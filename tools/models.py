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

class UnifiedCAE(nn.Module):
    """Image-only representation with two independent self-supervised signals.

    Steering is a prediction target, never an encoder input. Feeding the target
    into the latent (the legacy design) allowed the steering head to copy the
    answer and weakened image/action-consistency anomaly detection.
    """
    def __init__(self):
        super(UnifiedCAE, self).__init__()
        self.encoder = ImageEncoder()
        self.decoder = ImageDecoder(input_dim=256)
        self.steer_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        steer_pred = self.steer_head(latent)
        return x_recon, latent, steer_pred

class ControllerCNN(nn.Module):
    """
    NVIDIA-style CNN architecture for End-to-End Steering Angle Prediction.
    Input: (N, 3, 144, 224)
    Output: (N, 1) steering prediction
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.flatten_dim = 64 * 11 * 21
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError("ControllerCNN expects NCHW RGB tensors")
        x = x.float()
        x = x * 2.0 - 1.0

        feat = self.features(x)
        feat = feat.reshape(feat.size(0), -1)
        return self.classifier(feat)


class ControllerDeploymentWrapper(nn.Module):
    """Explicit car-runtime adapter: NHWC float pixels in [0, 255].

    Keeping deployment preprocessing outside ``ControllerCNN`` avoids tracing a
    data-dependent ``x.max()`` branch into ONNX. Training always uses torchvision
    NCHW tensors in [0, 1], while the existing car runtime contract remains NHWC
    pixels.
    """

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def forward(self, image):
        nchw = image.permute(0, 3, 1, 2).float() / 255.0
        return self.controller(nchw)

def calculate_pcc_tensor(img, output):
    b = img.shape[0]
    x = img.view(b, -1)
    y = output.view(b, -1)
    x_m = x - x.mean(dim=1, keepdim=True)
    y_m = y - y.mean(dim=1, keepdim=True)
    num = (x_m * y_m).sum(dim=1)
    den = torch.sqrt((x_m**2).sum(dim=1) * (y_m**2).sum(dim=1) + 1e-8)
    return (num / den).clamp(-1.0, 1.0)
