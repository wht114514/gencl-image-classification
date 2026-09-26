import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datetime import datetime

# ============ 参数 ============
EPOCHS = 10
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 基础的归一化（测试集统一用它）
normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# 三种训练数据亮度设置：
#   baseline : 原样
#   brighter : 亮度 x1.4（增加 40%）
#   darker   : 亮度 x0.6（减少 40%）
brightness_configs = {
    'baseline': 1.0,
    'brighter': 1.4,
    'darker':   0.6,
}

# 测试集固定用原始亮度
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)


class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


def make_trainloader(brightness):
    """根据亮度倍数构造训练集。ColorJitter 的 brightness 参数：
       brightness=1.0 表示不变；1.4 表示在 [0.6, 1.4] 范围内随机采样亮度。
       这里我们想精确控制为「固定 x1.4 / x0.6」，所以用 Lambda 直接缩放。"""
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
    """训练一个线性模型，返回每个 epoch 的测试准确率列表。"""
    trainloader = make_trainloader(brightness)
    net = LinearNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
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

        # 测试（固定原始亮度测试集）
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


# ============ 主流程：只跑变亮/变暗，基线直接用已有结果 ============
print(f'设备: {DEVICE}')

# 基线（原始亮度）不重跑，直接用你已测得的结果：10 个 epoch 均为 37.20%
BASELINE_FINAL_ACC = 37.20
results = {}

for name, brightness in brightness_configs.items():
    if name == 'baseline':
        continue
    print(f'\n开始训练 [{name}] 亮度 x{brightness}:')
    results[name] = train_one(brightness)

# ============ 画对比图 ============
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 6))
epochs = list(range(1, EPOCHS + 1))

colors = {'baseline': '#2E6FDB', 'brighter': '#E63946', 'darker': '#2A9D8F'}
labels = {'baseline': '基线(原始亮度)', 'brighter': '变亮 40%', 'darker': '变暗 40%'}
markers = {'baseline': 'o', 'brighter': 's', 'darker': '^'}

# 画变亮/变暗两条曲线
for name, accs in results.items():
    ax.plot(epochs, accs, color=colors[name], marker=markers[name],
            markersize=4, linewidth=2, label=labels[name])

# 基线画一条水平虚线（你已有的实验结果）
ax.axhline(BASELINE_FINAL_ACC, color=colors['baseline'], linestyle='--',
           linewidth=2, label=f"{labels['baseline']} {BASELINE_FINAL_ACC}%")

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('测试准确率 (%)', fontsize=13)
ax.set_ylim(0, 100)
ax.set_xlim(1, EPOCHS)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_title('训练集亮度变化对线性模型准确率的影响', fontsize=15,
             fontweight='bold', pad=15)
ax.legend(fontsize=11, framealpha=0.9)

fig.tight_layout()

filename = f'brightness_experiment_{datetime.now().strftime("%Y_%m_%d")}.png'
plt.savefig(filename, dpi=200, bbox_inches='tight')
print(f'\n对比图已保存为: {filename}')
plt.show()

# 打印总结
print('\n========== 实验结果 ==========')
print(f"{'基线(原始亮度)':>10}: 最终 {BASELINE_FINAL_ACC:.2f}%  (已有结果，未重跑)")
for name, accs in results.items():
    print(f'{labels[name]:>10}: 最终 {accs[-1]:.2f}%  (最高 {max(accs):.2f}%)')
