import src.model.resnet_imagenette as RNI
import src.helper.helper as helper
import src.xai.integrated_gradient as ig
import src.xai.grad_cam as gc
import src.xai.occlusion as oc
from torch import Tensor #th
from matplotlib import pyplot as plt
import warnings
###

# GuidedGradCam warnings
warnings.filterwarnings(
    "ignore",
    message="Setting backward hooks on ReLU activations"
)

def main() -> None:
    tds,vds,_,_ = RNI.create_datasets_dataloaders(RNI.Config)
    device = helper.get_device()
    _,resnet18 = helper.load_models_for_inference(RNI.Config.saved_pth_path, len(tds.classes), device)
    
    ### Get sample (ResNet18,Imagenette)
    raw_img:Tensor ; true_class:int #; idx:int
    raw_img,true_class,_ = helper.get_sample(vds,device)

    ### Get predicted class (ResNet18,Imagenette)
    predicted_class:int = helper.get_prediction(raw_img,resnet18)
    #print(f"predicted : {RNI.Config.class_names[vds.classes[predicted_class]]}")

    ### XAI methods
    ig_attr:Tensor = ig.get_integrated_gradient_attribute(raw_img,resnet18,predicted_class,device)
    gc_attr:Tensor = gc.get_guided_grad_cam_attribute(raw_img,resnet18,helper.get_last_conv_layer(resnet18),predicted_class)
    oc_attr:Tensor = oc.get_occlusion_attribute(raw_img,resnet18,predicted_class,device)

    ### Visualization
    img:Tensor = RNI.unnormalize_imagenette(raw_img[0].detach().cpu()).permute(1,2,0)
    ig_heat, ig_heat_min, ig_heat_max = ig.normalize_heatmap(ig_attr)
    gc_heat = gc.normalize_heatmap(gc_attr)
    oc_heat = oc.normalize_heatmap(oc_attr)

    plt.figure(figsize=(10,10))

    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{RNI.Config.class_names[vds.classes[predicted_class]]}/{RNI.Config.class_names[vds.classes[true_class]]}")

    plt.subplot(2,2,2)
    plt.imshow(img)
    plt.imshow(ig_heat, cmap="inferno",alpha=0.7,vmin=ig_heat_min,vmax=ig_heat_max,interpolation="nearest")
    plt.axis("off")
    plt.title("integrated gradient")

    plt.subplot(2,2,3)
    plt.imshow(img)
    plt.imshow(gc_heat, cmap="inferno",alpha=0.7,interpolation="nearest")
    plt.axis("off")
    plt.title("grad-cam")

    plt.subplot(2,2,4)
    plt.imshow(img)
    plt.imshow(oc_heat, cmap="inferno",alpha=0.5,interpolation="nearest")
    plt.axis("off")
    plt.title("occlusion")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()