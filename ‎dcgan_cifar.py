import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ============ 超参数 ============
EPOCHS = 50
BATCH_SIZE = 64
LATENT_DIM = 100
NUM_CLASSES = 10
LR = 0.0002
BETA1 = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)

# ============ 数据 ============
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=0)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


# ============ 条件生成器 cG ============
class Generator(nn.Module):
    """噪声 100 维 + 类别嵌入 100 维，相加后送入转置卷积。
    这样 G 可以“按指定类别”生成图片。"""
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
        z = z + self.label_emb(labels)          # 噪声 + 类别嵌入
        z = z.view(z.size(0), LATENT_DIM, 1, 1)
        return self.main(z)


# ============ 条件判别器 cD ============
class Discriminator(nn.Module):
    """图片 3 通道 + 类别嵌入图 1 通道，拼成 4 通道输入。"""
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, 32 * 32)
        self.main = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        lab = self.label_emb(labels).view(-1, 1, 32, 32)   # 类别 -> 1 通道图
        x = torch.cat([x, lab], dim=1)                     # 拼成 4 通道
        return self.main(x).view(-1, 1)


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)


netG = Generator().to(DEVICE)
netD = Discriminator().to(DEVICE)
netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

# 固定噪声 + 固定标签：每行一个类别（8 张/行），观察每类的生成质量
fixed_noise = torch.randn(64, LATENT_DIM, device=DEVICE)
fixed_labels = torch.arange(64, device=DEVICE) // 8   # 0..7 各 8 张

# ============ 进度图保存 ============
os.makedirs('gan_progress_cond', exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def save_progress(epoch):
    with torch.no_grad():
        fake = netG(fixed_noise, fixed_labels).detach().cpu()
    grid = vutils.make_grid(fake, nrow=8, normalize=True)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_title(f'Epoch {epoch+1}', fontsize=14)
    ax.imshow(grid.permute(1, 2, 0).numpy())
    fig.savefig(f'gan_progress_cond/epoch_{epoch+1:03d}.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


# ============ 训练 ============
print(f'设备: {DEVICE}')
G_losses, D_losses = [], []

for epoch in range(EPOCHS):
    for real_imgs, real_cls in dataloader:
        real_imgs, real_cls = real_imgs.to(DEVICE), real_cls.to(DEVICE)
        batch = real_imgs.size(0)

        real_labels = torch.full((batch, 1), 1.0, device=DEVICE)
        fake_labels = torch.full((batch, 1), 0.0, device=DEVICE)

        # ---- 1. 训练判别器 D（带类别条件）----
        netD.zero_grad()
        lossD_real = criterion(netD(real_imgs, real_cls), real_labels)

        noise = torch.randn(batch, LATENT_DIM, device=DEVICE)
        fake_cls = torch.randint(0, NUM_CLASSES, (batch,), device=DEVICE)
        fake_imgs = netG(noise, fake_cls)
        lossD_fake = criterion(netD(fake_imgs.detach(), fake_cls), fake_labels)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # ---- 2. 训练生成器 G ----
        netG.zero_grad()
        lossG = criterion(netD(fake_imgs, fake_cls), real_labels)
        lossG.backward()
        optimizerG.step()

        G_losses.append(lossG.item())
        D_losses.append(lossD.item())

    print(f'Epoch {epoch+1}/{EPOCHS}  Loss_D: {sum(D_losses[-len(dataloader):])/len(dataloader):.3f}'
          f'  Loss_G: {sum(G_losses[-len(dataloader):])/len(dataloader):.3f}')
    save_progress(epoch)

    if (epoch + 1) % 10 == 0:
        torch.save(netG.state_dict(), f'cdgan_generator_epoch{epoch+1}.pth')
        print(f'  已保存 cdgan_generator_epoch{epoch+1}.pth')

torch.save(netG.state_dict(), 'cdgan_generator_final.pth')
print('训练完成，已保存 cdgan_generator_final.pth')

# ============ loss 曲线 ============
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(G_losses, label='生成器 Loss', color='#E63946', alpha=0.7, linewidth=1)
ax.plot(D_losses, label='判别器 Loss', color='#2E6FDB', alpha=0.7, linewidth=1)
ax.set_xlabel('迭代次数 (batch)', fontsize=13)
ax.set_ylabel('Loss', fontsize=13)
ax.set_title('条件 DCGAN (cDCGAN) 训练 Loss 曲线', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.3)
fig.tight_layout()

filename = f'cdgan_loss_{datetime.now().strftime("%Y_%m_%d")}.png'
plt.savefig(filename, dpi=200, bbox_inches='tight')
print(f'Loss 曲线已保存为: {filename}')
plt.show()