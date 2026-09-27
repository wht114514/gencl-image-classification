import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datetime import datetime


EPOCHS = 10          
BATCH_SIZE = 64
LATENT_DIM = 100
NUM_CLASSES = 10
FAKE_PER_CLASS = 500  
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


train_transform = transforms.Compose([transforms.ToTensor(), normalize])
test_transform = train_transform

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)

real_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=train_transform)



class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, LATENT_DIM)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        z = z + self.label_emb(labels)
        z = z.view(z.size(0), LATENT_DIM, 1, 1)
        return self.main(z)


print(f'设备: {DEVICE}')
print('加载 cDCGAN 生成器...')
netG = Generator().to(DEVICE)
netG.load_state_dict(torch.load('cdgan_generator_final.pth', map_location=DEVICE))
netG.eval()

print(f'每类生成 {FAKE_PER_CLASS} 张，共 {FAKE_PER_CLASS * NUM_CLASSES} 张...')
fake_imgs_list, fake_labels_list = [], []
with torch.no_grad():
    for cls in range(NUM_CLASSES):
        # 分批生成，避免显存一次性占用过大
        for start in range(0, FAKE_PER_CLASS, 500):
            n = min(500, FAKE_PER_CLASS - start)
            noise = torch.randn(n, LATENT_DIM, device=DEVICE)
            labels = torch.full((n,), cls, dtype=torch.long, device=DEVICE)
            fake = netG(noise, labels)             # 输出 [-1,1]，与归一化后数据一致
            fake_imgs_list.append(fake.cpu())
            fake_labels_list.append(labels.cpu())

fake_imgs = torch.cat(fake_imgs_list)
fake_labels = torch.cat(fake_labels_list)
print(f'生成完成: {fake_imgs.shape[0]} 张假图')


#mix
from torch.utils.data import Dataset, ConcatDataset


class FakeDataset(Dataset):
    """自定义假图数据集：标签返回 int，和 CIFAR10 保持一致，
    这样 ConcatDataset 拼接时类型统一，collate 才不会报错。"""
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels.tolist()   # 转成 Python int 列表

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


fake_dataset = FakeDataset(fake_imgs, fake_labels)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train_and_eval(train_dataset, tag):
    """在给定训练集上训练 SimpleCNN，返回每个 epoch 的测试准确率。"""
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=0)
    net = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    accs = []
    for epoch in range(EPOCHS):
        net.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(inputs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        net.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                _, pred = torch.max(net(inputs), 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        accs.append(100 * correct / total)
        print(f'  [{tag}] Epoch {epoch+1}/{EPOCHS}  准确率: {accs[-1]:.2f}%')

    print(f'  [{tag}] 完成，最终 {accs[-1]:.2f}%')
    return accs



print('\n组 1: 纯真实数据')
accs_real = train_and_eval(real_trainset, '纯真实')

print('\n组 2: 真实 + GAN 生成数据')
mixed_dataset = ConcatDataset([real_trainset, fake_dataset])
accs_mixed = train_and_eval(mixed_dataset, '真实+生成')

#draw
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 6))
epochs = list(range(1, EPOCHS + 1))

ax.plot(epochs, accs_real, color='#2E6FDB', marker='o', markersize=4,
        linewidth=2, label=f'纯真实数据 (最终 {accs_real[-1]:.2f}%)')
ax.plot(epochs, accs_mixed, color='#E63946', marker='s', markersize=4,
        linewidth=2, label=f'真实+GAN生成 (最终 {accs_mixed[-1]:.2f}%)')

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('测试准确率 (%)', fontsize=13)
ax.set_ylim(0, 100)
ax.set_xlim(1, EPOCHS)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_title(f'GAN 生成数据对 SimpleCNN 的影响 (每类 +{FAKE_PER_CLASS} 张生成图)',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, framealpha=0.9)

fig.tight_layout()
filename = f'gan_augment_compare_{datetime.now().strftime("%Y_%m_%d")}.png'
plt.savefig(filename, dpi=200, bbox_inches='tight')
print(f'\n对比图已保存为: {filename}')
plt.show()

print('\n实验结论')
diff = accs_mixed[-1] - accs_real[-1]
print(f'纯真实数据:     {accs_real[-1]:.2f}%')
print(f'真实+GAN生成:   {accs_mixed[-1]:.2f}%')
print(f'差异: {diff:+.2f}%  ->  {"GAN 数据有提升" if diff > 0 else "GAN 数据无提升甚至有害" if diff < 0 else "基本持平"}')
