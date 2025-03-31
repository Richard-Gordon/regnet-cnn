# RegNet-CNN
PyTorch implementation of RegNet-like CNN models.

## Example: Model from name:
```
from regnet import models
model = models.get_model("RegNetX-200MF", num_classes=10)
```

Torchvision and Timm model name formats can also be used:
```
# All equivalent:
model_400mf             = models.get_model("RegNetX-400MF")
model_400mf_torchvision = models.get_model("regnet_x_400mf")
model_400mf_timm        = models.get_model("regnetx_004")
```

## Example: Model from parameters

```
from regnet import RegNetX
model_600mf = RegNetX(
    stage_depths     = [ 1,  3,   5,   7],
    stage_widths_out = [48, 96, 240, 528],
    stage_num_groups = [ 2,  4,  10,  22],
)
```

## Limitations
Only RegNetX is implemented.
Not on the Python Package Index (PyPI) yet.
