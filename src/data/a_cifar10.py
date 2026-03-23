"""
Docstring for data.a_cifar10
Program that process (automatically) cifar10 ~ torchvision.datasets.CIFAR10
"""

import torch
import torchvision
import torchvision.transforms
import warnings
from numpy.exceptions import VisibleDeprecationWarning

# cifar-10 uses a deprecated numpy version
warnings.filterwarnings(
    "ignore",
    category=VisibleDeprecationWarning,
)

# The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1].

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

def unnormalize_cifarTen(x : torch.Tensor):
    return (x * 0.5 + 0.5).clamp(0, 1)

ROOT: str = './cifar-10-torchvision'
BATCH_SIZE: int = 4
NUM_WORKERS: int = 2

def build_dataset(
        root: str, 
        train: bool, 
        transform: torchvision.transforms.Compose
    ) -> torchvision.datasets.CIFAR10:
    return torchvision.datasets.CIFAR10(
        root=root, 
        train=train, 
        download=True, 
        transform=transform
    )


def build_loader(
        dset: torch.utils.data.DataLoader, 
        batch_size: int, 
        shuffle: bool, 
        n_workers: int
    ) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)

trainset = build_dataset(ROOT, True, transform)

trainloader = build_loader(trainset, BATCH_SIZE, True, NUM_WORKERS)

testset = build_dataset(ROOT, False, transform)

testloader = build_loader(testset, BATCH_SIZE, False, NUM_WORKERS)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


##############################################################################################################


##### test, let us show some of the training images and their labels

# import matplotlib.pyplot as plt
# import numpy as np

# # functions to show an image


# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.savefig("plot.png", dpi=300, bbox_inches="tight")


# # get some random training images
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))