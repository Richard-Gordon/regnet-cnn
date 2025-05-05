import argparse
from pathlib import Path
from regnet.models import get_model, convert_model
import torch, torchvision
import torch.nn.functional as F
from torchvision.transforms._presets import ImageClassification
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


ARGUMENTS = [
    # Flag, Name             , Type , Default/Action, Help
    [ "-m", "--model-name"   , str  , 'regnetx_004' , "model name"            ],
    [ "-i", "--imagenet-path", str  , "./dataset"   , "imagenet dataset path" ],
    [ "-d", "--device"       , str  , "cuda"        , "device: e.g. cpu, cuda"],
    [ "-t", "--training"     , bool , 'store_true'  , "training mode"         ],
    [ "-e", "--num-epochs"   , int  , 1             , "number of epochs"      ],
    [ "-b", "--batch-size"   , int  , 250           , "batch size"            ],
    [ "-l", "--learning-rate", float, 0.1           , "SGD learning rate"     ],
    [ "-j", "--num-workers"  , int  , 8             , "number of  workers"    ],
]

def parse_args(**kwargs) -> argparse.Namespace:
    parser = argparse.ArgumentParser(**kwargs)

    for arg in ARGUMENTS:
        name_or_flags = [arg[0], arg[1]]
        kwargs = {'help': arg[4]}
        if type(arg_default:=arg[3]) == (arg_type:=arg[2]):
            kwargs['type'] = arg_type
            kwargs['default'] = arg_default
        else:
            kwargs['action'] = arg[3]
        parser.add_argument(*name_or_flags, **kwargs)

    return parser.parse_args()



def main(args):

    # Instantiate a pre-trained model from torchvision for testing
    if not args.training:
        model = torchvision.models.get_model(args.model_name, weights='DEFAULT')
        model = convert_model(model)
    # Otherwise, instantiate an untrained RegNet-X model for training
    else:
        model = get_model(args.model_name)

    model = model.to(args.device)

    # Vanilla SGD optimizer with default momentum and weight decay
    if args.training:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    # Imagenet dataset loaders with default transform (no augmentations)
    transform = ImageClassification(crop_size=224, resize_size=256)

    splits = {}
    for split in (['train', 'val'] if args.training else ['val']):

        training = split=='train'
        shuffle_data = training

        splits[split] = {'training': training}
        splits[split]['optimizer'] = optimizer if training else None

        split_path = Path(args.imagenet_path) / split
        splits[split]['dataloader'] = DataLoader(
            dataset     = ImageFolder(split_path, transform=transform),
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            shuffle     = shuffle_data,
        )
        
    
    # Training/testing loop
    for epoch in range(args.num_epochs):
        epoch_str = f'Epoch {epoch+1}/{args.num_epochs} '
        
        for split_name, split in splits.items():
            summary = run_one_epoch(
                model      = model,
                device     = args.device,
                dataloader = split['dataloader'],
                training   = split['training'],
                optimizer  = split['optimizer'],
                epoch_str  = epoch_str + f'({split_name}) | ',
            )
            if split_name == 'val':
                print(summary)


def run_one_epoch(model, device, dataloader, training, optimizer, epoch_str=''):
    """Run one epoch of training or testing"""
    model = model.train(training)
    summary = epoch_str

    num_images, num_correct, total_loss = 0,0,0
    for batch_index, (batch_images, batch_labels) in enumerate(dataloader):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        with torch.set_grad_enabled(training):
            batch_outputs = model(batch_images)
            loss = F.cross_entropy(batch_outputs, batch_labels)

        if training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        _, batch_predictions = torch.max(batch_outputs, dim=1)

        batch_size = batch_labels.size(0)
        num_images += batch_size
        total_loss += batch_size * loss.item()
        
        num_correct += (batch_predictions == batch_labels).sum().item()
        accuracy = num_correct / num_images

        # Print progress summary
        summary = epoch_str
        summary += f'Batch {batch_index+1}/{len(dataloader)} | '
        summary += f'Loss: {total_loss/num_images:.4f} | '
        summary += f'Accuracy: {accuracy:.2%} ({num_correct}/{num_images})'
        print(summary, end='\r')

        if num_images >= 50000:
            break

    # Clear the line after the last batch
    print(' '*len(summary), end='\r')
    return summary


if __name__ == "__main__":
    args = parse_args(description="RegNet-X Example Usage")
    main(args)
