# RegNet-CNN
PyTorch implementation of RegNet-like CNN models.

## Usage Examples

### Create a model instance by name
```
from regnet import models
model = models.get_model('RegNetX-200MF', num_classes=10)
```

Torchvision or Timm model naming conventions can also be used:
```
# All equivalent:
model_400mf = models.get_model('RegNetX-400MF')
model_400mf = models.get_model('regnet_x_400mf')
model_400mf = models.get_model('regnetx_004')
```

### Create a model instance directly using custom parameters

```
from regnet import RegNetX
model_600mf = RegNetX(
    stage_depths     = [ 1,  3,   5,   7],
    stage_widths_out = [48, 96, 240, 528],
    stage_num_groups = [ 2,  4,  10,  22],
)
```

### Create a model instance from another model - optionally copying the state

```
import torchvision, timm
from regnet import models

# Create a RegNetX instance matching an untrained model from Torchvision
torchvision_model = torchvision.models.get_model('regnet_x_800mf')
model_from_torchvision = models.convert_model(torchvision_model, state_transfer=False)

# Create a RegNetX instance matching a pre-trained model from Timm
timm_model = timm.create_model('regnetx_006', pretrained=True)
pretrained_model_from_timm = models.convert_model(timm_model, state_transfer=True)

```

## Limitations
Only RegNetX is implemented. \
Not on the Python Package Index (PyPI) yet.

## Attribution
The RegNet architecture was presented in the following [paper](https://arxiv.org/abs/2003.13678):
```
@misc{radosavovic2020designingnetworkdesignspaces,
      title={Designing Network Design Spaces}, 
      author={Ilija Radosavovic and Raj Prateek Kosaraju and Ross Girshick and Kaiming He and Piotr Dollár},
      year={2020},
      eprint={2003.13678},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2003.13678}, 
}
```
This project is not affiliated with, or endorsed by, the authors.

The model parameters were sourced from the [pycls repository](https://github.com/facebookresearch/pycls/tree/5719de641657c251f807c20a1fcf8ef6f4b60144/configs/dds_baselines/regnetx).
