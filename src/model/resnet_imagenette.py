from pathlib import Path
import src.helper.helper as helper
from torch import save, load, no_grad
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models.resnet import ResNet18_Weights
###


class Config:
    data_dir: Path = Path("imagenette2")
    class_names: dict[str, str] = {'n01440764' : 'tench fish', 'n02102040' : 'english springer', 'n02979186' : 'cassette player', 'n03000684' : 'chain saw', 'n03028079' : 'church', 'n03394916' : 'french horn', 'n03417042' : 'garbage truck', 'n03425413' : 'gas pump', 'n03445777' : 'golf ball', 'n03888257' : 'parachute'}
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-4
    epochs: int = 4
    saved_pth_path: Path = Path("resnet18_imagenette.pth")


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return train_transforms, eval_transforms

unnormalize_imagenette = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

def create_datasets_dataloaders(config: Config) -> tuple[datasets.ImageFolder, datasets.ImageFolder, DataLoader, DataLoader]:
    train_dir = config.data_dir / "train"
    val_dir = config.data_dir / "val"

    train_tfms, eval_tfms = get_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_tfms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def create_model(num_classes: int) -> models.ResNet:
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = Linear(in_features, num_classes)
    
    return model


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images) # prediction
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_correct / total


@no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_correct / total


def train_model(config: Config) -> None:
    device = helper.get_device()
    train_dataset, _, train_loader, val_loader = create_datasets_dataloaders(config)

    model = create_model(num_classes=len(train_dataset.classes)).to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.lr)

    print()
    print("Stating Training")
    print(f"Using device: {device}")
    print(f"Classes ID->name: {config.class_names}")
    print(f"Number of epochs: {config.epochs}")
    print()

    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

    save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, config.saved_pth_path)

    print(f"\nSaved checkpoint to {config.saved_pth_path}\n")


def load_model_for_inference(saved_pth_path, num_classes, device) -> models.ResNet:
    model = create_model(num_classes=num_classes).to(device)
    saved_model = load(saved_pth_path, map_location=device)
    model.load_state_dict(saved_model["model_state_dict"])
    model.eval()

    return model

# def load_model_for_training(saved_pth_path, model, optimizer : Adam, device) -> tuple[models.ResNet, torch.optim.Optimizer]:
#     saved_model = torch.load(saved_pth_path, map_location=device)
#     model.load_state_dict(saved_model["model_state_dict"])
#     optimizer.load_state_dict(saved_model["optimizer_state_dict"])
#     model.eval()

#     return model, optimizer


def main() -> None:
    config = Config()
    train_model(config)


if __name__ == "__main__":
    main()