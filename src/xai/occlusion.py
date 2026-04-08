"""
https://arxiv.org/pdf/1311.2901
Covers different parts of an input image with a square and observe how the class probability changes. 
This shows whether the model is relying on the actual object or just on background context.
"""
from captum.attr import Occlusion
import torch
from torch import Tensor, device, nn # type hints
import matplotlib.pyplot as plt

import src.helper.helper as helper
import src.model.resnet_imagenette as RNI #ResNetImagenette
#from src.data.a_cifar10 import testset as c10vds
###


class Config:
    sliding_window_shapes=(3, 16, 16), 
    strides=(3, 8, 8)
    ...

def normalize_heatmap(attr: Tensor) -> Tensor:
    """Return a the heat ratio"""
    heat = attr[0].clamp(min=0).sum(dim=0).detach().cpu()
    return (heat - heat.min()) / (heat.max() - heat.min())

def get_occlusion_attribute(
        x: Tensor, nn_model: nn.Module, prediction: int, device: device
    ) -> Tensor:
    baseline = torch.zeros_like(x).to(device)
    occlusion = Occlusion(nn_model)
    return occlusion.attribute(
        inputs=x, 
        target=prediction, 
        baselines=baseline, 
        sliding_window_shapes=Config.sliding_window_shapes,
        strides=Config.strides
    )


def main() -> None:
    tds, vds, _, _ = RNI.create_datasets_dataloaders(RNI.Config)
    
    device = helper.get_device()

    _, resnet18 = helper.load_models_for_inference(RNI.Config.saved_pth_path, len(tds.classes), device)

    ### Get sample
    ## LeNet, CIFAR-10
    #x, y_true, _ = helper.get_sample(c10vds, device)
    ## ResNet18, Imagenette
    a, b_true, _ = helper.get_sample(vds, device)

    ### Chose target (predicted class)
    ## LeNet, CIFAR-10
    #y_pred: int = helper.get_prediction(x, convnet1)
    ## 2
    b_pred: int = helper.get_prediction(a, resnet18)
    #print(b_pred, b_true)


    ##### Occlusion
    attr: Tensor = get_occlusion_attribute(a, resnet18, b_pred, device)


    ### Visualization
    ## ResNet18, Imagenette
    img: Tensor = RNI.unnormalize_imagenette(a[0].detach().cpu()).clamp(0,1).permute(1,2,0)
    heat: Tensor = normalize_heatmap(attr)

    plt.figure(figsize=(8,4), dpi=200)

    plt.subplot(1,2,1)
    plt.imshow(img, interpolation="nearest")
    plt.axis("off")
    plt.title(f"Original={RNI.Config.class_names[vds.classes[b_true]]}")

    plt.subplot(1,2,2)
    plt.imshow(img, interpolation="nearest")
    plt.imshow(heat, cmap="inferno", alpha=0.45, interpolation="nearest")
    plt.axis("off")
    plt.title(f"Predicted={RNI.Config.class_names[vds.classes[b_pred]]}")

    plt.show()


if __name__ == "__main__":
    main()