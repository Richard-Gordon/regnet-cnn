from torch import nn
from typing import Type
from functools import partial

class RegNetX(nn.Module):

    def __init__(self,
        stage_depths    : list[int] = [],
        stage_widths_out: list[int] = [],
        stage_widths_mid: list[int] = None,
        stage_num_groups: list[int] = None,
        stem_width      : int = 32,
        num_classes     : int = 1000,
    ):
        """CNN model based on RegNetX family from arXiv:2003.13678v1

        Arguments:
            stage_depths     : Depth of each stage
            stage_widths_out : Output width of each stage
            stage_widths_mid : Middle width of each stage
                - Bottleneck if mid < out, expansion if mid > out
                - If None then widths_mid = widths_out
            stage_num_groups : Number of groups for each stage
                - N.B. not group widths
                - If None then depth-wise, i.e. num_groups = widths_mid
            stem_width       : Width of the stem layer (default 32)
            num_classes      : Number of output classes (default 1000)
        """
        super().__init__()

        stages_ic = [stem_width] + stage_widths_out[:-1]
        stages_mc = stage_widths_mid or stage_widths_out
        stages_oc = stage_widths_out

        # Depthwise separable convolutions if groups aren't defined
        stages_groups = stage_num_groups or stages_mc

        stages_kwargs = [
            {'ic': ic, 'mc': mc, 'oc': oc, 'groups': groups, 'depth': depth}
                for ic, mc, oc, groups, depth in zip(
                    stages_ic,
                    stages_mc,
                    stages_oc,
                    stages_groups,
                    stage_depths,
            )
        ]

        self.stem = RegNetLayer(ic=3, oc=stem_width, ks=3, stride=2)

        self.body = self.stages = nn.Sequential(
            *[RegNetStage(**stage_kwargs) for stage_kwargs in stages_kwargs]
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(stages_oc[-1], num_classes),
        )


    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)
        return x



class RegNetStage(nn.Module):

    def __init__(self,
        ic: int, mc: int, oc: int,
        groups: int = 1,
        depth : int = 1,
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            RegNetBlock(ic, mc, oc, groups, stride=2), *[
            RegNetBlock(oc, mc, oc, groups, stride=1) for _ in range(depth-1)
        ])


    def forward(self, x):
        return self.blocks(x)



class RegNetBlock(nn.Module):

    def __init__(self,
        ic: int, mc: int, oc: int,
        groups: int = 1,
        stride: int = 1,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            RegNetLayer(ic, mc, ks=1),
            RegNetLayer(mc, mc, ks=3, groups=groups, stride=stride),
            RegNetLayer(mc, oc, ks=1, act_type = nn.Identity),
        )

        if stride==1:
            self.residual = nn.Identity()
        else:
            self.residual = RegNetLayer(
                ic, oc, ks=1,
                act_type = nn.Identity,
                stride = stride
            )

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        fx = self.layers(x)
        rx = self.residual(x)
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

        conv_kwargs['padding'] = conv_kwargs.get('padding', ks//2)
        conv_kwargs['bias']    = conv_kwargs.get('bias', False)

        self.conv = nn.Conv2d(ic, oc, ks, **conv_kwargs)
        self.norm = nn.BatchNorm2d(oc)
        self.act  = act_type()


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
