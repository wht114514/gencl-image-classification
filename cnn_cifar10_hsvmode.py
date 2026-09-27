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


def rgb_to_hsv(img_tensor):
   
    from PIL import Image
    import torchvision.transforms.functional as F
    img_pil = F.to_pil_image(img_tensor)      # (3,H,W) -> PIL RGB
    img_hsv = img_pil.convert('HSV')          # PIL HSV
    return F.to_tensor(img_hsv)               # (3,H,W) 张量，范围 [0,1]



test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(rgb_to_hsv),
    normalize,
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(rgb_to_hsv),
    normalize,
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)


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


print(f'设备: {DEVICE}')
net = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_losses = []
test_accs = []

for epoch in range(EPOCHS):
    net.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(net(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    train_losses.append(avg_loss)
    scheduler.step()

    net.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            _, pred = torch.max(net(inputs), 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    acc = 100 * correct / total
    test_accs.append(acc)
    print(f'Epoch {epoch+1}/{EPOCHS}  Loss: {avg_loss:.3f}  准确率: {acc:.2f}%')


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 6))
epochs = list(range(1, EPOCHS + 1))


ax.plot(epochs, test_accs, color='#2A9D8F', marker='^', markersize=4,
        linewidth=2, label='HSV 颜色空间')


BASELINE_ACC = 72.87
ax.axhline(BASELINE_ACC, color='#2E6FDB', linestyle='--', linewidth=2,
           label=f'RGB 基线 {BASELINE_ACC}%')

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('测试准确率 (%)', fontsize=13)
ax.set_ylim(0, 100)
ax.set_xlim(1, EPOCHS)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_title('颜色空间 RGB vs HSV 对 CNN 准确率的影响', fontsize=15,
             fontweight='bold', pad=15)
ax.legend(fontsize=11, framealpha=0.9)

fig.tight_layout()

filename = f'hsv_experiment_{datetime.now().strftime("%Y_%m_%d")}.png'
plt.savefig(filename, dpi=200, bbox_inches='tight')
print(f'\n对比图已保存为: {filename}')
plt.show()

print('\n实验结果')
print(f'RGB 基线: {BASELINE_ACC:.2f}%')
print(f'HSV 实验: 最终 {test_accs[-1]:.2f}%  (最高 {max(test_accs):.2f}%)')
