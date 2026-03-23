import torch
import training.train_cifar10 as engine
import src.data.a_cifar10 as ds
from matplotlib.pyplot import imshow
import torchvision
import src.model.cnn_cifar10 as cnn


def main() -> None:
    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # print(f"Using {device}")

    # print("||||| TRAINING |||||")
    # TRAIN_TIMES: int = 10
    # engine.train_and_save_model(TRAIN_TIMES)
    # print("")

    print("||||| TESTING |||||")
    # ground-truth / prediction
    dataiter = iter(ds.testloader)
    images, labels = next(dataiter)
    grid = torchvision.utils.make_grid(images)
    grid = grid.permute(1, 2, 0).clamp(0, 1) # CHW -> HWC
    imshow(grid)
    print("GroundTruth: ", " ".join(f"{ds.classes[labels[j]]:5s}" for j in range(4)))
    ConvNet = cnn.ConvNet1()
    ConvNet.load_state_dict(torch.load("./cifar_net.pth", weights_only=True))
    outputs = ConvNet(images)
    _, predicted = torch.max(outputs, 1)
    print("Predicted: ", " ".join(f"{ds.classes[predicted[j]]:5s}" for j in range(4)))

    # accuracy for each class
    correct_prediction = {classname: 0 for classname in ds.classes}
    total_prediction = {classname: 0 for classname in ds.classes}
    with torch.no_grad():
        for data in ds.testloader:
            images, labels = data
            outputs = ConvNet(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_prediction[ds.classes[label]] += 1
                total_prediction[ds.classes[label]] += 1
    
    for classname, correct_count in correct_prediction.items():
        accuracy = 100 * float(correct_count) / total_prediction[classname]
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f}%")


if __name__ == "__main__":
    main()