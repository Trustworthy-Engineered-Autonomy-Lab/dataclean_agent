import torch
import torch.nn as nn
from torchvision import transforms
from .image_contract import IMAGE_HEIGHT, IMAGE_WIDTH

TRANSFORM_224_224 = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
])

_ENCODED_HEIGHT = IMAGE_HEIGHT // 8
_ENCODED_WIDTH = IMAGE_WIDTH // 8

class ImageEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * _ENCODED_HEIGHT * _ENCODED_WIDTH, 256), nn.ReLU(True)
        )

class ImageDecoder(nn.Sequential):
    def __init__(self, input_dim=256):
        super().__init__(
            nn.Linear(input_dim, 64 * _ENCODED_HEIGHT * _ENCODED_WIDTH), nn.ReLU(True),
            nn.Unflatten(1, (64, _ENCODED_HEIGHT, _ENCODED_WIDTH)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

class SteerEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(1, 64), nn.ReLU(True),
            nn.Linear(64, 128), nn.ReLU(True),
            nn.Linear(128, 256), nn.ReLU(True),
        )


class IROS2026CAE(nn.Module):
    """IROS2026 action-conditioned CAE, adapted to RGB 224x224.

    Three image convolution stages; the image and steering embeddings are
    averaged before reconstruction. There is no steering prediction head/loss.
    """
    def __init__(self):
        super().__init__()
        self.encoder = ImageEncoder()
        self.steer_encoder = SteerEncoder()
        self.decoder = ImageDecoder(input_dim=256)

    def forward(self, x, steering):
        if x.dim() != 4 or tuple(x.shape[1:]) != (3, IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(
                f"IROS2026CAE expects NCHW RGB tensors shaped "
                f"[N,3,{IMAGE_HEIGHT},{IMAGE_WIDTH}]"
            )
        if tuple(steering.shape) != (x.shape[0], 1):
            raise ValueError("IROS2026CAE expects steering shaped [N, 1]")
        latent = (self.encoder(x) + self.steer_encoder(steering)) / 2.0
        x_recon = self.decoder(latent)
        return x_recon, latent

class ControllerCNN(nn.Module):
    """
    NVIDIA-style CNN architecture for End-to-End Steering Angle Prediction.
    Input: (N, 3, 224, 224)
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
        with torch.no_grad():
            probe = torch.zeros(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
            self.flatten_dim = int(self.features(probe).reshape(1, -1).size(1))
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
        if tuple(x.shape[-2:]) != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(
                f"ControllerCNN expects spatial size {IMAGE_HEIGHT}x{IMAGE_WIDTH}"
            )
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
    x = img.reshape(b, -1)
    y = output.reshape(b, -1)
    x_m = x - x.mean(dim=1, keepdim=True)
    y_m = y - y.mean(dim=1, keepdim=True)
    num = (x_m * y_m).sum(dim=1)
    # Match eval_epoch in the supplied IROS source: epsilon AFTER norm product.
    den = torch.norm(x_m, p=2, dim=1) * torch.norm(y_m, p=2, dim=1) + 1e-8
    return (num / den).clamp(-1.0, 1.0)
