from captum.attr import GuidedGradCam
from torch import Tensor, nn
from matplotlib import pyplot as plt

import src.model.resnet_imagenette as RNI #ResNetImagenette
import src.helper.helper as helper
###

def get_guided_grad_cam_attribute(
        x: Tensor, nn_model: nn.Module, last_layer: nn.Module, target: int
    ) -> Tensor:
    guided_grad_cam = GuidedGradCam(nn_model, last_layer)
    return guided_grad_cam.attribute(inputs=x, target=target)

def get_last_conv_layer(nn_model: nn.Module) -> nn.Conv2d:
    last_conv_layer = None
    for module in nn_model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv_layer = module
    return last_conv_layer

def normalize_heatmap(attr: Tensor) -> Tensor:
    heat = attr[0].sum(dim=0).clamp(min=0).detach().cpu()
    return (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

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
    last_conv_layer: nn.Conv2d = get_last_conv_layer(resnet18)

    ##### Grad-Cam
    attr: Tensor = get_guided_grad_cam_attribute(a, resnet18, last_conv_layer, b_pred)

    ### Visualization
    img: Tensor = RNI.unnormalize_imagenette(a[0].detach().cpu()).permute(1,2,0)
    heat: Tensor = normalize_heatmap(attr)

    plt.figure(figsize=(10,5), dpi=200)

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"original={RNI.Config.class_names[vds.classes[b_true]]}")

    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.imshow(heat, cmap="inferno", alpha=0.6, interpolation="nearest")
    #plt.colorbar()
    plt.axis("off")
    plt.title(f"predicted={RNI.Config.class_names[vds.classes[b_pred]]}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()