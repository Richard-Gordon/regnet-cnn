from regnet.regnetx import RegNetX
from regnet.params import get_model_params

def get_model_kwargs(name: str) -> dict:
    """Load model constructor keyword arguments"""
    model_params = get_model_params(name)

    stage_depths     = model_params['di']
    stage_widths_out = model_params['wi']
    stage_widths_mid = [w *  model_params["b"] for w in stage_widths_out]
    stage_num_groups = [w // model_params['g'] for w in stage_widths_mid]

    model_params = {
        'stem_width'      : 32,
        'stage_depths'    : stage_depths,
        'stage_widths_out': stage_widths_out,
        'stage_widths_mid': stage_widths_mid,
        'stage_num_groups': stage_num_groups,
    }
    return model_params


def get_model(name: str, **config) -> RegNetX:
    """Get a model instance by name"""
    model_params = get_model_kwargs(name)
    return RegNetX(**model_params, **config)
