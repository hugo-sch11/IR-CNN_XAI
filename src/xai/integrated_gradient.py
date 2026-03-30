"""
https://arxiv.org/pdf/1703.01365
Compare l'entrée à la baseline, intègre les gradients de la baseline à l'entrée.
Donne un taux d'activation par entrée.
Propriété : Sensibilité, Complétitude (Completeness)
(point positif comparé au gradient de base)
+ évite la saturation des gradients = plus stable
+ contribution total au lieu d'une sensibilité local
- choix baseline, couteux en calcul
- assume un straight-line path (tout au long du gradient rester dans la même distribution de donnée, alors qu'en réalité dans le cnn, des images produites sont hors du répertoire de donnée), ce qui donne des attributions trompeuses ou instables (appuyé par un gradient d'un artifact = fmap)
"""
from captum.attr import IntegratedGradients
import torch
from torch import Tensor, device, nn # type hints
import matplotlib.pyplot as plt

import src.helper.helper as helper
import src.model.ln_cifar10.a_cifar10 as ds1
import src.model.resnet_imagenette as RNI #ResNetImagenette
###


class Config:
    n_steps: int = 64
    ...

def normalize_heatmap(attr: Tensor) -> Tensor:
    heat = attr[0].clamp(min=0).sum(dim=0).detach().cpu()
    return (heat - heat.min()) / (heat.max() - heat.min())

def get_integrated_gradient_attribute(
        x: Tensor, nn_model: nn.Module, prediction: int, device: device
    ) -> Tensor:
    IntegratedGradient = IntegratedGradients(nn_model)
    baseline: Tensor = torch.zeros_like(x).to(device)
    return IntegratedGradient.attribute(
        inputs=x, 
        baselines=baseline, 
        target=prediction, 
        n_steps=Config.n_steps
    )


def main() -> None:
    tds, vds, tdl, vdl = RNI.create_datasets_dataloaders(RNI.Config)

    device = helper.get_device()

    ConvNet1, resnet18 = helper.load_models_for_inference(RNI.Config.saved_pth_path, len(tds.classes), device)

    ### Get sample
    ## LeNet, CIFAR-10
    x: Tensor; y_true: int # img from ds, class index
    x, y_true, _ = helper.get_sample(ds1.testset, device)
    ## ResNet18, Imagenette
    a: Tensor; b_true: int # img from ds, class index
    a, b_true, _ = helper.get_sample(vds, device)

    ### Get target (predicted class)
    ## LeNet, CIFAR-10
    y_pred: int = helper.get_prediction(x, ConvNet1)
    ## ResNet18, Imagenette
    b_pred: int = helper.get_prediction(a, resnet18)
    #print(b_pred, b_true)


    ##### Integrated Gradients
    attr1: Tensor = get_integrated_gradient_attribute(x, ConvNet1, y_pred, device)
    attr2: Tensor = get_integrated_gradient_attribute(a, resnet18, b_pred, device)
    #print(attr1.shape) #-> torch.Size([1, 3, 32, 32])
    #print(attr2.shape) #-> torch.Size([1, 3, 224, 224])
    #-> (N,C,H,W) -> (batch_size=image_processed_together, channels=color_channels, height, width)


    img1: Tensor = ds1.unnormalize_cifarTen(x[0].detach().cpu()).permute(1, 2, 0)
    img2: Tensor = RNI.unnormalize_imagenette(a[0].detach().cpu()).clamp(0, 1).permute(1, 2, 0)

    heat1: Tensor = normalize_heatmap(attr1)
    heat2: Tensor = normalize_heatmap(attr2)

    ### Visualization
    ## LeNet, CIFAR-10
    plt.figure(figsize=(8,4), dpi=200)

    plt.subplot(1,2,1)
    plt.imshow(img1, interpolation="nearest")
    plt.axis("off")
    plt.title(f"Original={ds1.classes[y_true]}")

    plt.subplot(1,2,2)
    plt.imshow(img1, interpolation="nearest")
    plt.imshow(heat1, cmap="inferno", alpha=0.45, interpolation="nearest")
    plt.axis("off")
    plt.title(f"Predicted={ds1.classes[y_pred]}")
    
    plt.show()

    ## ResNet18, Imagenette
    plt.figure(figsize=(8,4), dpi=200)

    plt.subplot(1,2,1)
    plt.imshow(img2, interpolation="nearest")
    plt.axis("off")
    plt.title(f"Original={RNI.Config.class_names[vds.classes[b_true]]}")

    plt.subplot(1,2,2)
    plt.imshow(img2, interpolation="nearest")
    plt.imshow(heat2, cmap="inferno", alpha=0.45, interpolation="nearest")
    plt.axis("off")
    plt.title(f"Predicted={RNI.Config.class_names[vds.classes[b_pred]]}")
    
    plt.show()

if __name__ == "__main__":
    main()