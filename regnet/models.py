from copy import deepcopy
from collections import defaultdict
from torch import nn
from regnet.regnetx import RegNetX
from regnet.params import get_model_params


def get_model(name: str, **config) -> RegNetX:
    """Get a model instance by name"""
    kwargs = get_model_kwargs(name)
    model = RegNetX(**kwargs, **config)
    return model


def get_model_kwargs(name: str) -> dict:
    """Get model constructor keyword arguments by name"""
    params = get_model_params(name)

    stage_widths_out = params['wi']
    stage_widths_mid = [width *  params['b'] for width in stage_widths_out]
    stage_num_groups = [width // params['g'] for width in stage_widths_mid]

    kwargs = {
        'stem_kwargs'     : params['stem_kwargs'],
        'head_kwargs'     : params['head_kwargs'],
        'stage_depths'    : params['di'],
        'stage_widths_out': stage_widths_out,
        'stage_widths_mid': stage_widths_mid,
        'stage_num_groups': stage_num_groups,
    }
    return kwargs


is_subclass = lambda obj, *types: issubclass(obj.__class__, types)

def extract_model_kwargs(model: nn.Module) -> dict:
    kwargs = defaultdict(list)
    stage_depth = 0

    # Iterate over all modules that are Conv2d or Linear
    for module in model.modules():

        # Find the channel widths from the convolution layers
        if is_subclass(module, nn.Conv2d):

            # Record the input width of the stem (preceding any RegNet stages)
            if 'stem_kwargs' not in kwargs:
                kwargs['stem_kwargs'] = {
                    'ic'    : module.in_channels,
                    'oc'    : module.out_channels,
                    'ks'    : module.kernel_size[0],
                    'stride': module.stride[0],
                }

            # Select the spatial convolutions first
            elif module.kernel_size > (1,1):

                # Each stage starts with a downscaling spatial convolution
                if module.stride > (1,1):
                    # Record the expansion/bottleneck width and number of groups
                    kwargs['stage_widths_mid'].append(module.out_channels)
                    kwargs['stage_num_groups'].append(module.groups)

                    # Record the depth of the previous stage, if there was one
                    if stage_depth > 0:
                        kwargs['stage_depths'].append(stage_depth)
                        stage_depth = 0

                # Number of spatial convolutions between downscaling layers
                stage_depth += 1

            # Output width, after any expansion/bottleneck
            elif len(kwargs['stage_widths_out']) < \
                 len(kwargs['stage_widths_mid']):
                kwargs['stage_widths_out'].append(module.out_channels)

        # Record the number of classes from the final linear layer
        elif is_subclass(module, nn.Linear):
            kwargs['head_kwargs'] = {'out_features': module.out_features}

    # The final stage is completed implicitly
    kwargs['stage_depths'].append(stage_depth)
    return kwargs


def get_branch(module: nn.Module, branch: str = None):
    """Get the branch of the model that the module belongs to"""
    if is_subclass(module, nn.Conv2d, nn.BatchNorm2d, nn.Linear):
        if (
            is_subclass(module, nn.Conv2d) and
            module.kernel_size==(1,1) and
            module.stride==(2,2)
        ):
            branch = 'residual'
        elif not (branch=='residual' and is_subclass(module, nn.BatchNorm2d)):
            branch = 'main'
        return branch
    return None


def copy_branch_state(model: nn.Module) -> dict:
    """Copy the state dict of the model, preserving the branch structure"""
    branch_state = defaultdict(list)
    branch = None
    for module in model.modules():
        branch = get_branch(module, branch)
        if branch is not None:
            state_dict = deepcopy(module.state_dict())
            branch_state[branch].append(state_dict)

    return branch_state


def load_branch_state(model: nn.Module, branch_state: dict) -> nn.Module:
    """Load the state dict into the model, preserving the branch structure"""
    branch = None
    for module in model.modules():
        branch = get_branch(module, branch)
        if branch in branch_state:
            state_dict = branch_state[branch].pop(0)
            module.load_state_dict(state_dict)
    return model


def transfer_state(src_model: nn.Module, dst_model: nn.Module) -> nn.Module:
    """Transfer the state dict from one model to another"""
    src_branch_state = copy_branch_state(src_model)
    dst_model = load_branch_state(dst_model, src_branch_state)
    return dst_model


def convert_model(model: nn.Module, state_transfer=True, **config) -> RegNetX:
    """Get a model instance matching the configuration of an existing model"""
    kwargs = extract_model_kwargs(model)
    dst_model = RegNetX(**kwargs, **config)
    if state_transfer:
        dst_model = transfer_state(model, dst_model)
    return dst_model
