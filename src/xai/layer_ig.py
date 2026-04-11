from captum.attr import LayerIntegratedGradients
import torch
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from torch import Tensor, nn, device
import torch.nn.functional as F

import src.model.resnet_imagenette as RNI #ResNetImagenette
import src.helper.helper as helper
###


def get_layer_integrated_gradients_attribute(
        x: Tensor, nn_model: nn.Module, last_layer: nn.Module, target: int, device: device
    ) -> Tensor:
    layer_integrated_gradients = LayerIntegratedGradients(nn_model, last_layer)
    #baseline: Tensor = torch.zeros_like(x).to(device)
    baseline = torch.tensor(gaussian_filter(x.detach().cpu().numpy(), sigma=3)).to(device)
    return layer_integrated_gradients.attribute(
        inputs=x,
        baselines=baseline,
        target=target,
        n_steps=256
    )

def normalize_heatmap(attr: Tensor, target_size: tuple) -> Tensor:
    heat = attr[0].sum(dim=0).detach().cpu()
    # upsample
    heat = F.interpolate(heat.unsqueeze(0).unsqueeze(0), size=target_size).squeeze()
    heat = helper.smooth_heatmap(heat)
    return heat / (heat.abs().max() + 1e-8) # [-1,1]

def main() -> None:
    tds, vds, _, _ = RNI.create_datasets_dataloaders(RNI.Config)
    device = helper.get_device()
    _, resnet18 = helper.load_models_for_inference(RNI.Config.saved_pth_path, len(tds.classes), device)

    ### Get sample
    a: Tensor; b_true: int # img, class_idx
    a, b_true, _ = helper.get_sample(vds, device)

    ### Get target
    b_pred: int = helper.get_prediction(a, resnet18)

    ### Get last convolutional layer
    last_conv_layer: nn.Conv2d = helper.get_last_conv_layer(resnet18)

    ##### Layer Integrated Gradients
    attr: Tensor = get_layer_integrated_gradients_attribute(a, resnet18, last_conv_layer, b_pred, device)

    ### Visualization
    img: Tensor = RNI.unnormalize_imagenette(a[0].detach().cpu()).permute(1,2,0)
    heat: Tensor = normalize_heatmap(attr, target_size=(img.shape[0],img.shape[1])) # (224,224)

    plt.figure(figsize=(10,5), dpi=200)

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"original={RNI.Config.class_names[vds.classes[b_true]]}")

    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.imshow(heat, cmap="seismic", alpha=0.5, vmin=-1.0, vmax=1.0, interpolation="nearest")
    #plt.colorbar()
    plt.axis("off")
    plt.title(f"predicted={RNI.Config.class_names[vds.classes[b_pred]]}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()