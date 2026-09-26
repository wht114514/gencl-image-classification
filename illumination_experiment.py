import os
import argparse
import torch
import torch.nn as nn
from PIL import ImageOps
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.amp import autocast
from model import get_model


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, nargs='+',
                        default=['./resnet50_best.pth', './resnet50_bestmax.pth'])
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--data_path', type=str, default='./dataset/')
    return parser


class BrightnessAdjust:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        return TF.adjust_brightness(img, self.factor)


class HistogramEqualize:
    def __call__(self, img):
        return ImageOps.equalize(img)


def make_transform(brightness=None, equalize=False):
    ops = []
    if brightness is not None:
        ops.append(BrightnessAdjust(brightness))
    if equalize:
        ops.append(HistogramEqualize())
    ops += [
        transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)


CONDITIONS = [
    ('normal',          None, False),
    ('brightness x0.4', 0.4,  False),
    ('brightness x1.6', 1.6,  False),
    ('x0.4 + hist-eq',  0.4,  True),
    ('x1.6 + hist-eq',  1.6,  True),
]


@torch.no_grad()
def evaluate_one(model, device, data_path, batch_size, brightness, equalize):
    transform = make_transform(brightness, equalize)
    dataset = datasets.CIFAR10(root=data_path, train=False, download=False,
                               transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    correct = 0
    total = 0
    for images, labels in tqdm(loader, file=os.sys.stdout, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(images)
        preds = torch.max(outputs, 1)[1]
        correct += torch.eq(preds, labels).sum().item()
        total += labels.size(0)
    return correct / total


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Use device:', device)

    results = {}
    for weights_path in args.weights:
        model = get_model(args.model)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print(f'\n=== {os.path.basename(weights_path)} ===')
        results[weights_path] = {}
        for name, brightness, equalize in CONDITIONS:
            acc = evaluate_one(model, device, args.data_path,
                               args.batch_size, brightness, equalize)
            results[weights_path][name] = acc
            print('{:<18} {:.4f}'.format(name, acc))

    print('\n' + '=' * 60)
    header = '{:<18}'.format('condition')
    for weights_path in args.weights:
        header += '{:>22}'.format(os.path.basename(weights_path))
    print(header)
    print('-' * 60)
    for name, _, _ in CONDITIONS:
        row = '{:<18}'.format(name)
        for weights_path in args.weights:
            row += '{:>22.4f}'.format(results[weights_path][name])
        print(row)
    print('=' * 60)


if __name__ == '__main__':
    main(get_argparse().parse_args())
