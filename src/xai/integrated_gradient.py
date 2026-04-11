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
#import src.model.ln_cifar10.a_cifar10 as ds1
import src.model.resnet_imagenette as RNI #ResNetImagenette
###


def get_integrated_gradient_attribute(
        x: Tensor, nn_model: nn.Module, prediction: int, device: device
    ) -> Tensor:
    IntegratedGradient = IntegratedGradients(nn_model)
    #baseline: Tensor = torch.zeros_like(x).to(device) * 0.5
    baseline = torch.tensor(gaussian_filter(x.detach().cpu().numpy(), sigma=3)).to(device)
    return IntegratedGradient.attribute(
        inputs=x, 
        baselines=baseline, 
        target=prediction, 
        n_steps=256
    )

def normalize_heatmap(attr: Tensor) -> tuple[Tensor,float,float]:
    heat = attr[0].abs().sum(dim=0).detach().cpu()
    heat = helper.smooth_heatmap(heat)
    return (heat - heat.min()) / (heat.max() - heat.min() + 1e-8), 0, 1.0 # [0,1]


def main() -> None:
    tds, vds, _, _ = RNI.create_datasets_dataloaders(RNI.Config)

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
    attr: Tensor = get_integrated_gradient_attribute(a, resnet18, b_pred, device)
    #print(attr2.shape) #-> torch.Size([1, 3, 224, 224])
    #-> (N,C,H,W) -> (batch_size=image_processed_together, channels=color_channels, height, width)

    ### Visualization
    ## ResNet18, Imagenette
    img: Tensor = RNI.unnormalize_imagenette(a[0].detach().cpu()).permute(1,2,0)
    heat, heat_min, heat_max = normalize_heatmap(attr)

    plt.figure(figsize=(10,5), dpi=200)

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"original={RNI.Config.class_names[vds.classes[b_true]]}")

    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.imshow(heat, cmap="seismic", alpha=0.5, vmin=heat_min, vmax=heat_max, interpolation="nearest")
    #plt.colorbar()
    plt.axis("off")
    plt.title(f"predicted={RNI.Config.class_names[vds.classes[b_pred]]}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()