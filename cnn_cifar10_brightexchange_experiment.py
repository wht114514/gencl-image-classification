import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datetime import datetime


EPOCHS = 10
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

#亮度设置
brightness_configs = {
    'baseline': 1.0,
    'brighter': 1.4,
    'darker':   0.6,
}


test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                 # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                 # 16x16 -> 8x8
        )
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def make_trainloader(brightness):

    if brightness == 1.0:
        train_transform = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        def scale_brightness(img_tensor):
            return torch.clamp(img_tensor * brightness, 0.0, 1.0)
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(scale_brightness),
            normalize,
        ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    return torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                       shuffle=True, num_workers=0)


def train_one(brightness):

    trainloader = make_trainloader(brightness)
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

    print(f'  亮度 x{brightness:.1f} 完成，最终准确率 {accs[-1]:.2f}%')
    return accs



print(f'设备: {DEVICE}')

# 基线72.87%
BASELINE_FINAL_ACC = 72.87
results = {}

for name, brightness in brightness_configs.items():
    if name == 'baseline':
        continue
    print(f'\n开始训练 [{name}] 亮度 x{brightness}:')
    results[name] = train_one(brightness)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 6))
epochs = list(range(1, EPOCHS + 1))

colors = {'baseline': '#2E6FDB', 'brighter': '#E63946', 'darker': '#2A9D8F'}
labels = {'baseline': '基线(原始亮度)', 'brighter': '变亮 40%', 'darker': '变暗 40%'}
markers = {'baseline': 'o', 'brighter': 's', 'darker': '^'}


for name, accs in results.items():
    ax.plot(epochs, accs, color=colors[name], marker=markers[name],
            markersize=4, linewidth=2, label=labels[name])


ax.axhline(BASELINE_FINAL_ACC, color=colors['baseline'], linestyle='--',
           linewidth=2, label=f"{labels['baseline']} {BASELINE_FINAL_ACC}%")

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('测试准确率 (%)', fontsize=13)
ax.set_ylim(0, 100)
ax.set_xlim(1, EPOCHS)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_title('训练集亮度变化对 CNN 准确率的影响', fontsize=15,
             fontweight='bold', pad=15)
ax.legend(fontsize=11, framealpha=0.9)

fig.tight_layout()

filename = f'brightness_experiment_cnn_{datetime.now().strftime("%Y_%m_%d")}.png'
plt.savefig(filename, dpi=200, bbox_inches='tight')
print(f'\n对比图已保存为: {filename}')
plt.show()

# 打印总结
print('\n实验结果')
print(f"{'基线(原始亮度)':>10}: 最终 {BASELINE_FINAL_ACC:.2f}%  (已有结果，未重跑)")
for name, accs in results.items():
    print(f'{labels[name]:>10}: 最终 {accs[-1]:.2f}%  (最高 {max(accs):.2f}%)')
