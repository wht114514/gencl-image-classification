import os
import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.amp import autocast
from model import get_model


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./checkpoint/resnet50_best.pth')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--data_path', type=str, default='./dataset/')
    parser.add_argument('--class_indices', type=str, default='./class_indices.json')
    return parser


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Use device:', device)

    transform_test = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.CIFAR10(root=args.data_path, train=False,
                                    download=False, transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4,
                             pin_memory=True, persistent_workers=True)

    with open(args.class_indices, 'r') as f:
        class_indices = json.load(f)
    num_classes = len(class_indices)

    model = get_model(args.model)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    state_dict = torch.load(args.weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f'Loaded weights from {args.weights}')
    print(f'Number of test images: {len(test_dataset)}')

    correct = 0
    total = 0
    class_correct = {name: 0 for name in class_indices.values()}
    class_total = {name: 0 for name in class_indices.values()}

    test_bar = tqdm(test_loader, file=os.sys.stdout)
    with autocast('cuda', dtype=torch.bfloat16):
        for images, labels in test_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            preds = torch.max(outputs, 1)[1]

            correct += torch.eq(preds, labels).sum().item()
            total += labels.size(0)

            for pred, label in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                name = class_indices[str(label)]
                class_total[name] += 1
                if pred == label:
                    class_correct[name] += 1

            test_bar.desc = "test accuracy: {:.4f}".format(correct / total)

    accuracy = correct / total
    print()
    print("=" * 40)
    print("Overall accuracy: {:.4f}".format(accuracy))
    print("=" * 40)
    print("Per-class accuracy:")
    for name in class_indices.values():
        acc = class_correct[name] / class_total[name] if class_total[name] > 0 else 0
        print("  {:<12}: {:.4f} ({}/{})".format(name, acc, class_correct[name], class_total[name]))

    return accuracy


if __name__ == '__main__':
    args = get_argparse().parse_args()
    evaluate(args)
