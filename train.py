
import os
import argparse
import sys
import torch.nn as nn
import torch
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import json
import torch.utils.data as Data
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import *
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast
def get_argparse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=20,help='number of epochs')
    parser.add_argument('--batch_size',type=int,default=400,help='batch size')
    parser.add_argument('--data_path',type=str,default='./dataset/',help='path to dataset')
    parser.add_argument('--model',type=str,default='resnet50',help='model name')
    parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
    parser.add_argument('--save_dir',type=str,default='./checkpoint/',help='save .pth')
    parser.add_argument('--num_classes',type=int,default=10,help='number of classes')
    parser.add_argument('--resume_weights', type=str, default=None, help='path to .pth to fine-tune from')
    parser.add_argument('--best_acc_init', type=float, default=0, help='resume 的历史最佳，避免 best 被低分覆盖')
    parser.add_argument('--resume_epoch', type=int, default=0, help='起始 epoch')

    return parser


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Use device:', device)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(160, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])


    train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=False, transform=transform_train)
    val_dataset = datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform_val)
    val_num = len(val_dataset)
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    class_dict = dict((val,key) for key,val in flower_list.items())

    json_str = json.dumps(class_dict,indent=4)
    with open('class_indices.json','w') as json_file:
        json_file.write(json_str)

    num_workers = 12#min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print("Using batch_size={} dataloader worker every process.".format(num_workers))

    train_loader = Data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=num_workers,pin_memory=True,persistent_workers=True)
    val_loader = Data.DataLoader(val_dataset,batch_size=args.batch_size,num_workers=num_workers,shuffle=False,pin_memory=True,persistent_workers=True)

    model = get_model(args.model)
    num_ftrs = model.fc.in_features 
    model.fc = torch.nn.Linear(num_ftrs, len(flower_list))
    model = model.cuda()
    
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)


    #params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = optim.Adam(params,args.lr)

    fc_params   = [p for p in model.fc.parameters()]
    backbone_params = [p for n, p in model.named_parameters()
                        if not n.startswith('fc.')]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1, 'weight_decay': 0.01}, 
        {'params': fc_params,      'lr': args.lr, 'weight_decay': 0.01},
        ])


    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    batch_num = len(train_loader)
    total_time = 0
    best_acc = args.best_acc_init
    if args.resume_weights:
        state_dict = torch.load(args.resume_weights, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        print(f'Loaded weights from {args.resume_weights}, best_acc so far: {args.best_acc_init}')
    #model = torch.compile(model)
    for epoch in range(args.resume_epoch, args.epochs):

        start_time = time.perf_counter()

        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader,file=sys.stdout)

        for step,data in enumerate(train_bar):
            train_images,train_labels = data

            train_images = train_images.to(device, non_blocking=True)
            train_labels = train_labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(train_images)
                loss = loss_function(outputs, train_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss: {:.3f}".format(epoch+1,args.epochs,loss)

        model.eval()
        val_acc = 0
        var_bar = tqdm(val_loader,file=sys.stdout)

        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            for val_data in var_bar:
                val_images, val_labels = val_data
                val_images = val_images.to(device,non_blocking=True)
                val_labels = val_labels.to(device,non_blocking=True)

                val_y=model(val_images)
                pred_y = torch.max(val_y,1)[1]

                val_acc += torch.eq(pred_y,val_labels).sum().item()

                var_bar.desc = "val eopch[{}/{}]".format(epoch+1,args.epochs)

        val_accurate = val_acc / val_num
        current_lr = optimizer.param_groups[1]['lr']
        print("[epoch {:.0f}] lr: {:.6f} train_loss: {:.3f} val_accuracy: {:.3f}".format(
            epoch+1, current_lr, train_loss/batch_num, val_accurate))

        epoch_time = time.perf_counter()-start_time
        print("epoch_time:{}".format(epoch_time))
        total_time += epoch_time
        scheduler.step()
        print()

        if val_accurate > best_acc:
            best_acc = val_accurate

            torch.save(model.state_dict(), os.path.join(args.save_dir, args.model+'_best.pth'))



    m,s = divmod(total_time,60)
    h,m = divmod(m,60)
    print("total time:{:0f}:{:0f}:{:0f}".format(h,m,s))
    print("Finished Training!")




if __name__ == '__main__':
    args = get_argparse().parse_args()
    train(args)