"""
https://arxiv.org/pdf/1703.01365
Compare l'entrée à la baseline, intègre les gradients de la baseline à l'entrée.
Donne un taux d'activation par entrée.
Propriété : Sensibilité, Complétude (Completeness)
(point positif comparé au gradient de base)
+ évite la saturation des gradients = plus stable
+ contribution total au lieu d'une sensibilité local
- choix baseline, couteux en calcul
- assume un straight-line path (tout au long du gradient rester dans la même distribution de donnée, alors qu'en réalité dans le cnn, des images produites sont hors du répertoire de donnée), ce qui donne des attributions trompeuses ou instables (appuyé par un gradient d'un artifact = fmap)
"""
from captum.attr import IntegratedGradients
import torch
from scipy.ndimage import gaussian_filter
from torch import Tensor, device, nn # type hints
import matplotlib.pyplot as plt

import src.helper.helper as helper
import src.model.ln_cifar10.a_cifar10 as ds1
import src.model.resnet_imagenette as RNI #ResNetImagenette
###


class Config:
    n_steps: int = 128
    ...

def get_integrated_gradient_attribute(
        x: Tensor, nn_model: nn.Module, prediction: int, device: device
    ) -> Tensor:
    IntegratedGradient = IntegratedGradients(nn_model)
    #baseline: Tensor = torch.zeros_like(x).to(device) * 0.5
    baseline = torch.tensor(gaussian_filter(x.detach().cpu().numpy(), sigma=10)).to(device)
    return IntegratedGradient.attribute(
        inputs=x, 
        baselines=baseline, 
        target=prediction, 
        n_steps=Config.n_steps
    )

def normalize_heatmap(attr: Tensor) -> Tensor:
    heat = attr[0].sum(dim=0).detach().cpu()
    heat = smooth_heatmap(heat)
    #print(heat.min(), heat.max(), heat.mean())
    return heat / (heat.abs().max() + 1e-8)

def smooth_heatmap(heat: Tensor, sigma: float = 0.5) -> Tensor:
    smoothed = gaussian_filter(heat.numpy(), sigma=sigma)
    return torch.tensor(smoothed)


def main() -> None:
    tds, vds, tdl, vdl = RNI.create_datasets_dataloaders(RNI.Config)

    device = helper.get_device()

    _, resnet18 = helper.load_models_for_inference(RNI.Config.saved_pth_path, len(tds.classes), device)

    ### Get sample
    ## ResNet18, Imagenette
    a: Tensor; b_true: int # img from ds, class index
    a, b_true, _ = helper.get_sample(vds, device)

    ### Get target (predicted class)
    ## ResNet18, Imagenette
    b_pred: int = helper.get_prediction(a, resnet18)
    #print(b_pred, b_true)


    ##### Integrated Gradients
    attr2: Tensor = get_integrated_gradient_attribute(a, resnet18, b_pred, device)
    #print(attr2.shape) #-> torch.Size([1, 3, 224, 224])
    #-> (N,C,H,W) -> (batch_size=image_processed_together, channels=color_channels, height, width)

    img2: Tensor = RNI.unnormalize_imagenette(a[0].detach().cpu()).permute(1,2,0)

    heat2 = normalize_heatmap(attr2)
    #print(f"min {min2},  max {max2}")

    ### Visualization
    ## ResNet18, Imagenette
    plt.figure(figsize=(10,5), dpi=200)

    plt.subplot(1,2,1)
    plt.imshow(img2)
    plt.axis("off")
    plt.title(f"original={RNI.Config.class_names[vds.classes[b_true]]}")

    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.imshow(heat2, cmap="seismic", alpha=0.6, vmin=-1, vmax=1, interpolation="nearest")
    #plt.colorbar()
    plt.axis("off")
    plt.title(f"predicted={RNI.Config.class_names[vds.classes[b_pred]]}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()