import torch
import torch.nn as nn
from torchvision import transforms

TRANSFORM_144_224 = transforms.Compose([
    transforms.Resize((144, 224)),
    transforms.ToTensor(),
])

class ImageEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128 * 9 * 14, 256), nn.ReLU(True)
        )

class ImageDecoder(nn.Sequential):
    def __init__(self, input_dim=256):
        super().__init__(
            nn.Linear(input_dim, 128 * 9 * 14), nn.ReLU(True),
            nn.Unflatten(1, (128, 9, 14)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

class SteerEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(1, 64), nn.ReLU(True),
            nn.Linear(64, 128), nn.ReLU(True),
            nn.Linear(128, 256), nn.ReLU(True)
        )

class UnifiedCAE(nn.Module):
    def __init__(self):
        super(UnifiedCAE, self).__init__()
        self.encoder = ImageEncoder()
        self.decoder = ImageDecoder(input_dim=256)
        self.steer_encoder = SteerEncoder()
        self.steer_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, x, steer):
        img_latent = self.encoder(x)
        z_act = self.steer_encoder(steer)
        latent = (img_latent + z_act) / 2.0
        x_recon = self.decoder(latent)
        steer_pred = self.steer_head(latent)
        return x_recon, latent, steer_pred

def calculate_pcc_tensor(img, output):
    b = img.shape[0]
    x = img.view(b, -1)
    y = output.view(b, -1)
    x_m = x - x.mean(dim=1, keepdim=True)
    y_m = y - y.mean(dim=1, keepdim=True)
    num = (x_m * y_m).sum(dim=1)
    den = torch.sqrt((x_m**2).sum(dim=1) * (y_m**2).sum(dim=1) + 1e-8)
    return (num / den).clamp(-1.0, 1.0)