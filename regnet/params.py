# Parameters for RegNetX models from the paper: arXiv:2003.13678v1
regnetx_params = {
    'RegNetX-200MF': {'di': [1, 1,  4,  7], 'wi': [ 24,  56,  152,  368], 'g':   8, 'b': 1, 'e': 30.8, 'wa': 36, 'w0':  24, 'wm': 2.5},
    'RegNetX-400MF': {'di': [1, 2,  7, 12], 'wi': [ 32,  64,  160,  384], 'g':  16, 'b': 1, 'e': 27.2, 'wa': 24, 'w0':  24, 'wm': 2.5},
    'RegNetX-600MF': {'di': [1, 3,  5,  7], 'wi': [ 48,  96,  240,  528], 'g':  24, 'b': 1, 'e': 25.5, 'wa': 37, 'w0':  48, 'wm': 2.2},
    'RegNetX-800MF': {'di': [1, 3,  7,  5], 'wi': [ 64, 128,  288,  672], 'g':  16, 'b': 1, 'e': 24.8, 'wa': 36, 'w0':  56, 'wm': 2.3},
    'RegNetX-1.6GF': {'di': [2, 4, 10,  2], 'wi': [ 72, 168,  408,  912], 'g':  24, 'b': 1, 'e': 22.9, 'wa': 34, 'w0':  80, 'wm': 2.2},
    'RegNetX-3.2GF': {'di': [2, 6, 15,  2], 'wi': [ 96, 192,  432, 1008], 'g':  40, 'b': 1, 'e': 21.6, 'wa': 26, 'w0':  88, 'wm': 2.2},
    'RegNetX-4.0GF': {'di': [2, 5, 14,  2], 'wi': [ 80, 240,  560, 1360], 'g':  48, 'b': 1, 'e': 21.3, 'wa': 39, 'w0':  96, 'wm': 2.4},
    'RegNetX-6.4GF': {'di': [2, 4, 10,  1], 'wi': [168, 392,  784, 1624], 'g':  56, 'b': 1, 'e': 20.7, 'wa': 61, 'w0': 184, 'wm': 2.1},
    'RegNetX-8.0GF': {'di': [2, 5, 15,  1], 'wi': [ 80, 240,  720, 1920], 'g': 120, 'b': 1, 'e': 20.5, 'wa': 50, 'w0':  80, 'wm': 2.9},
    'RegNetX-12GF' : {'di': [2, 5, 11,  1], 'wi': [224, 448,  896, 2240], 'g': 112, 'b': 1, 'e': 20.3, 'wa': 73, 'w0': 168, 'wm': 2.4},
    'RegNetX-16GF' : {'di': [2, 6, 13,  1], 'wi': [256, 512,  896, 2048], 'g': 128, 'b': 1, 'e': 20.0, 'wa': 56, 'w0': 216, 'wm': 2.1},
    'RegNetX-32GF' : {'di': [2, 7, 13,  1], 'wi': [336, 672, 1344, 2520], 'g': 168, 'b': 1, 'e': 19.5, 'wa': 70, 'w0': 320, 'wm': 2.0},
}

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
    return params
