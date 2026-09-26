import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datetime import datetime

# ---------- 1. 数据 ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=0)
testloader  = torch.utils.data.DataLoader(testset, batch_size=64,
                                          shuffle=False, num_workers=0)

# ---------- 2. 模型：CNN ----------
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = SimpleCNN().to(device)
print(f'使用设备: {device}')

# ---------- 3. 损失 / 优化器 / 学习率调度 ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

EPOCHS = 50
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------- 4. 训练 + 记录 ----------
train_losses = []
test_accs = []

for epoch in range(EPOCHS):
    # 训练
    net.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    train_losses.append(avg_loss)
    scheduler.step()

    # 测试
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    test_accs.append(acc)
    print(f'Epoch {epoch+1}/{EPOCHS}  Loss: {avg_loss:.3f}  测试准确率: {acc:.2f}%')

# ---------- 5. 画曲线（美化版） ----------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示为方块

fig, ax1 = plt.subplots(figsize=(10, 6))
epochs = list(range(1, EPOCHS + 1))

# 左轴：训练 Loss
color_loss = '#2E6FDB'
ax1.plot(epochs, train_losses, color=color_loss, linewidth=2,
         marker='o', markersize=4, label='训练 Loss')
ax1.set_xlabel('Epoch', fontsize=13)
ax1.set_ylabel('Loss', color=color_loss, fontsize=13)
ax1.tick_params(axis='y', labelcolor=color_loss, labelsize=10)
ax1.tick_params(axis='x', labelsize=10)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_xlim(1, EPOCHS)

# 右轴：测试准确率
color_acc = '#E63946'
ax2 = ax1.twinx()
ax2.plot(epochs, test_accs, color=color_acc, linewidth=2,
         marker='s', markersize=4, label='测试准确率')
ax2.set_ylabel('Accuracy (%)', color=color_acc, fontsize=13)
ax2.tick_params(axis='y', labelcolor=color_acc, labelsize=10)
ax2.set_ylim(0, 100)

# 标注最终准确率
best_acc = max(test_accs)
final_acc = test_accs[-1]
ax2.axhline(best_acc, color=color_acc, linestyle=':', linewidth=1, alpha=0.5)
ax2.annotate(f'最高 {best_acc:.2f}%',
             xy=(test_accs.index(best_acc) + 1, best_acc),
             xytext=(test_accs.index(best_acc) + 1 - 8, best_acc + 6),
             fontsize=10, color=color_acc,
             arrowprops=dict(arrowstyle='->', color=color_acc, lw=1))

plt.title('Simple CNN on CIFAR-10 训练曲线', fontsize=15, fontweight='bold', pad=15)

# 合并两个轴的图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10,
           framealpha=0.9)

fig.tight_layout()

# 根据当前时间保存图像，命名规则：netname_yyyy_mm_dd.png
netname = 'cnn'
date_str = datetime.now().strftime('%Y_%m_%d')
filename = f'{netname}_{date_str}.png'
plt.savefig(filename, dpi=200, bbox_inches='tight')
print(f'曲线图已保存为: {filename}')

plt.show()

print(f'\n最终测试准确率: {test_accs[-1]:.2f}%  (最高: {best_acc:.2f}%)')
