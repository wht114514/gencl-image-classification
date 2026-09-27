import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from datetime import datetime

LATENT_DIM = 100
CHANNELS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Generator(nn.Module):
    def __init__(self):
        super().__init__()
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
            nn.ConvTranspose2d(64, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), LATENT_DIM, 1, 1)
        return self.main(z)



checkpoint = 'generator_final.pth'
netG = Generator().to(DEVICE)
netG.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
netG.eval()


with torch.no_grad():
    noise = torch.randn(64, LATENT_DIM, device=DEVICE)
    fake_imgs = netG(noise).cpu()

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(10, 10))
plt.axis('off')
plt.title(f'DCGAN 生成的 CIFAR-10 图片 ({checkpoint})', fontsize=14, fontweight='bold')
# 从 [-1,1] 还原到 [0,1]
grid = vutils.make_grid(fake_imgs, nrow=8, normalize=True)
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.tight_layout()

filename = f'dcgan_samples_{datetime.now().strftime("%Y_%m_%d")}.png'
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f'生成图已保存为: {filename}')
plt.show()
