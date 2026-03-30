"""helpers"""
import torch
import random
import src.model.ln_cifar10.cnn_cifar10 as cnn1
import src.model.resnet_imagenette as RNI #ResNetImagenette
# type hints
from torch import device, Tensor, nn
from torchvision.models import ResNet
from torchvision.datasets import VisionDataset


def get_device() -> device:
    return device("cuda" if torch.cuda.is_available() else "cpu")


def load_models_for_inference(
        path: str, len_dsclasses: int, device: device
    ) -> tuple[cnn1.ConvNet1, ResNet]:
    ## LeNet~
    ConvNet1 = cnn1.ConvNet1().to(device)
    ConvNet1.load_state_dict(torch.load("./cifar_net.pth", map_location=device))
    ConvNet1.eval() # inference mode
    ## ResNet18
    resnet18 = RNI.load_model_for_inference(path, len_dsclasses, device)
    return ConvNet1, resnet18


def get_sample(ds: VisionDataset, device: device) -> tuple[Tensor, int, int]:
    """Return a random img and its class index and the random index used."""
    idx = random.randrange(len(ds))
    x: Tensor; y: int
    x, y = ds[idx]
    x = x.unsqueeze(0).to(device).requires_grad_(True) # required for captum.attr
    return x, y, idx


def get_prediction(x: Tensor, nn_model: nn.Module) -> int:
    """Return the class index prediction of the model from the input."""
    logits: Tensor = nn_model(x)
    return logits.argmax(dim=1).item()

# TODO: plot helper