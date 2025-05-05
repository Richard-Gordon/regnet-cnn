from torch import nn
from typing import Type
from functools import partial


class RegNetX(nn.Module):

    def __init__(self,
        stem_kwargs     : dict = {'ic': 3, 'oc': 32, 'ks': 3, 'stride': 2},
        stage_depths    : list[int] = [],
        stage_widths_out: list[int] = [],
        stage_widths_mid: list[int] = None,
        stage_num_groups: list[int] = None,
        head_kwargs     : dict = {'out_features': 1000},
    ):
        """CNN model based on RegNetX family from arXiv:2003.13678v1

        Arguments:
            stem_kwargs      : Keyword arguments for the stem layer
                - ic    : Input  channels (default: 3)
                - oc    : Output channels (default: 32)
                - ks    : Kernel size     (default: 3)
                - stride: Stride          (default: 2)
            stage_depths     : Depth of each stage
            stage_widths_out : Output width of each stage
            stage_widths_mid : Middle width of each stage
                - Bottleneck if mid < out, expansion if mid > out
                - If None then widths_mid = widths_out
            stage_num_groups : Number of groups for each stage
                - N.B. not group widths
                - If None then depth-wise, i.e. num_groups = widths_mid
            head_kwargs      : Keyword arguments for the classification head
                - out_features: Number of output features
        """
        super().__init__()

        # Prepend stem_width to output widths to form input widths
        stage_widths_inp = [stem_kwargs['oc']] + stage_widths_out[:-1]
        # No expansion/bottleneck if mid-widths are not defined
        stage_widths_mid = stage_widths_mid or stage_widths_out
        # Depthwise separable convolutions if groups aren't defined
        stage_groups     = stage_num_groups or stage_widths_mid

        # Convert module arguments into per-stage keyword-argument dictionaries
        stages_configs = {
            'ic'    : stage_widths_inp,
            'mc'    : stage_widths_mid,
            'oc'    : stage_widths_out,
            'depth' : stage_depths,
            'groups': stage_groups,
        }
        satges_keywords = stages_configs.keys()
        stages_args = zip(*stages_configs.values())

        stages_kwargs = [
            dict(zip(satges_keywords, stage_args)) for stage_args in stages_args
        ]

        # Instantiate the network: stem, body, and classification head
        self.stem = RegNetLayer(**stem_kwargs)

        self.body = self.stages = nn.Sequential(
            *[RegNetStage(**stage_kwargs) for stage_kwargs in stages_kwargs]
        )

        # Classifier head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(stage_widths_out[-1], **head_kwargs),
        )


    def forward(self, x):
        stem_out = self.stem(x)
        body_out = self.body(stem_out)
        head_out = self.head(body_out)
        return head_out



class RegNetStage(nn.Module):

    def __init__(self,
        ic: int, mc: int, oc: int,
        depth : int = 1,
        groups: int = 1,
    ):
        super().__init__()

        # Initial layer with channel upsampling, and spatial downscaling
        block_0 = partial(RegNetBlock, ic, mc, oc, groups=groups, stride=2)
        # Remaining layers
        block_n = partial(RegNetBlock, oc, mc, oc, groups=groups, stride=1)

        self.blocks = nn.Sequential(
            block_0(), *[
            block_n() for _ in range(depth-1)
        ])

    def forward(self, x):
        return self.blocks(x)



class RegNetBlock(nn.Module):

    def __init__(self,
        ic: int, mc: int, oc: int,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()

        if stride==1:
            self.residual = nn.Identity()
        else:
            self.residual = RegNetLayer(
                ic, oc, ks=1,
                act_type = nn.Identity,
                stride = stride
            )

        self.layers = nn.Sequential(
            RegNetLayer(ic, mc, ks=1),
            RegNetLayer(mc, mc, ks=3, stride=stride, groups=groups),
            RegNetLayer(mc, oc, ks=1, act_type=nn.Identity),
        )

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        rx = self.residual(x)
        fx = self.layers(x)
        x = fx + rx
        x = self.relu(x)
        return x



InplaceReLU = partial(nn.ReLU, inplace=True)

class RegNetLayer(nn.Module):

    def __init__(self,
        ic: int, oc: int, ks: int,
        act_type: Type[nn.Module] = InplaceReLU,
        **conv_kwargs
    ):
        super().__init__()

        # Set custom default Conv2d arguments, overridden by module inputs
        conv_kwargs['padding'] = conv_kwargs.get('padding', (ks-1)//2)
        conv_kwargs['bias'   ] = conv_kwargs.get('bias'   , False)

        self.conv = nn.Conv2d(ic, oc, ks, **conv_kwargs)
        self.norm = nn.BatchNorm2d(oc)
        self.act  = act_type()


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
