# Parameters for RegNetX models defined by the paper: arXiv:2003.13678v1
#   d = depth, g = group-width, b = bottleneck-factor,
#   w0 = initial-width, wa = width-slope, wm = width-multiplier
regnetx_params = {
    'RegNetX-200MF': {'d':13, 'w0': 24, 'wa':36.44, 'wm':2.49, 'g':  8, 'b':1 },
    'RegNetX-400MF': {'d':22, 'w0': 24, 'wa':24.48, 'wm':2.54, 'g': 16, 'b':1 },
    'RegNetX-600MF': {'d':16, 'w0': 48, 'wa':36.97, 'wm':2.24, 'g': 24, 'b':1 },
    'RegNetX-800MF': {'d':16, 'w0': 56, 'wa':35.73, 'wm':2.28, 'g': 16, 'b':1 },
    'RegNetX-1.6GF': {'d':18, 'w0': 80, 'wa':34.01, 'wm':2.25, 'g': 24, 'b':1 },
    'RegNetX-3.2GF': {'d':25, 'w0': 88, 'wa':26.31, 'wm':2.25, 'g': 48, 'b':1 },
    'RegNetX-4.0GF': {'d':23, 'w0': 96, 'wa':38.65, 'wm':2.43, 'g': 40, 'b':1 },
    'RegNetX-6.4GF': {'d':17, 'w0':184, 'wa':60.83, 'wm':2.07, 'g': 56, 'b':1 },
    'RegNetX-8.0GF': {'d':23, 'w0': 80, 'wa':49.56, 'wm':2.88, 'g':120, 'b':1 },
    'RegNetX-12GF' : {'d':19, 'w0':168, 'wa':73.36, 'wm':2.37, 'g':112, 'b':1 },
    'RegNetX-16GF' : {'d':22, 'w0':216, 'wa':55.59, 'wm':2.10, 'g':128, 'b':1 },
    'RegNetX-32GF' : {'d':23, 'w0':320, 'wa':69.86, 'wm':2.00, 'g':168, 'b':1 },
}

# Derived parameters for RegNetX models: di = stage-depths, wi = stage-widths
regnetx_derived = {
    'RegNetX-200MF': { 'di': [1, 1,  4,  7], 'wi': [ 24,  56,  152,  368] },
    'RegNetX-400MF': { 'di': [1, 2,  7, 12], 'wi': [ 32,  64,  160,  384] },
    'RegNetX-600MF': { 'di': [1, 3,  5,  7], 'wi': [ 48,  96,  240,  528] },
    'RegNetX-800MF': { 'di': [1, 3,  7,  5], 'wi': [ 64, 128,  288,  672] },
    'RegNetX-1.6GF': { 'di': [2, 4, 10,  2], 'wi': [ 72, 168,  408,  912] },
    'RegNetX-3.2GF': { 'di': [2, 6, 15,  2], 'wi': [ 96, 192,  432, 1008] },
    'RegNetX-4.0GF': { 'di': [2, 5, 14,  2], 'wi': [ 80, 240,  560, 1360] },
    'RegNetX-6.4GF': { 'di': [2, 4, 10,  1], 'wi': [168, 392,  784, 1624] },
    'RegNetX-8.0GF': { 'di': [2, 5, 15,  1], 'wi': [ 80, 240,  720, 1920] },
    'RegNetX-12GF' : { 'di': [2, 5, 11,  1], 'wi': [224, 448,  896, 2240] },
    'RegNetX-16GF' : { 'di': [2, 6, 13,  1], 'wi': [256, 512,  896, 2048] },
    'RegNetX-32GF' : { 'di': [2, 7, 13,  1], 'wi': [336, 672, 1344, 2520] },
}

# Merge the derived parameters with the base parameters
for name, params in regnetx_derived.items():
    regnetx_params[name].update(params)


# Torchvision library model names
models_aliases_torchvision = {
    'regnet_x_200mf': 'RegNetX-200MF',
    'regnet_x_400mf': 'RegNetX-400MF',
    'regnet_x_600mf': 'RegNetX-600MF',
    'regnet_x_800mf': 'RegNetX-800MF',
    'regnet_x_1_6gf': 'RegNetX-1.6GF',
    'regnet_x_3_2gf': 'RegNetX-3.2GF',
    'regnet_x_4_0gf': 'RegNetX-4.0GF',
    'regnet_x_6_4gf': 'RegNetX-6.4GF',
    'regnet_x_8gf'  : 'RegNetX-8.0GF',
    'regnet_x_12gf' : 'RegNetX-12GF',
    'regnet_x_16gf' : 'RegNetX-16GF',
    'regnet_x_32gf' : 'RegNetX-32GF',
}

# Timm library model names
models_aliases_timm = {
    'regnetx_002': 'RegNetX-200MF',
    'regnetx_004': 'RegNetX-400MF',
    'regnetx_006': 'RegNetX-600MF',
    'regnetx_008': 'RegNetX-800MF',
    'regnetx_016': 'RegNetX-1.6GF',
    'regnetx_032': 'RegNetX-3.2GF',
    'regnetx_040': 'RegNetX-4.0GF',
    'regnetx_064': 'RegNetX-6.4GF',
    'regnetx_080': 'RegNetX-8.0GF',
    'regnetx_120': 'RegNetX-12GF',
    'regnetx_160': 'RegNetX-16GF',
    'regnetx_320': 'RegNetX-32GF',
}

models_aliases = models_aliases_torchvision | models_aliases_timm


def get_model_params(name: str) -> dict:
    """Get model parameters for the given model name"""
    name = models_aliases.get(name, name)
    params = regnetx_params[name]
    params['stem_kwargs'] = {'ic': 3, 'oc': 32, 'ks': 3, 'stride': 2}
    params['head_kwargs'] = {'out_features': 1000}
    return params
