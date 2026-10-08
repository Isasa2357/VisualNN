# train VGG16, VGG32
# dataset: Z:\dataset\imagenette2-160
# train set: Z:\dataset\imagenette2-160\train. ex. "Z:\dataset\imagenette2-160\train\n01440764\ILSVRC2012_val_00000293.JPEG"
# val set: Z:\dataset\imagenette2-160\val. ex. "Z:\dataset\imagenette2-160\val\n01440764\ILSVRC2012_val_00009111.JPEG"

# 呼び出し方
# python main.py 
#   --dataset [dataset, default: Z:\dataset\imagenette2-160]
#   --epochs [number of epochs, default: 100]
#   --batch-size [batch size, default: 32]
#   --lr [learning rate, default: 0.001]
#   --device [device to use, default: cuda if available else cpu]
#   --class_num [number of classes, default: 10]
#   --project [project name, default: project]
#   --result [path to save result, default: result]

import argparse
import os
import csv

import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from .vgg import VGG16

def eval(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    '''
    モデルを評価する

    Args:
        model (nn.Module): 評価するモデル
        dataloader (DataLoader): 評価用データローダ

    Returns:
        float: 精度
    '''
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", position=1):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total if total > 0 else 0.0

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int, lr: float) -> list[tuple[float, float]]:
    '''
    モデルを学習する

    Args:
        model (nn.Module): 学習するモデル
        train_loader (DataLoader): 学習用データローダ
        val_loader (DataLoader): 評価用データローダ
        device (torch.device): デバイス
        epochs (int): エポック数
        lr (float): 学習率

    Returns:
        list[tuple[float, float]]: 各エポックの(学習精度, 評価精度)
    '''
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    accs = []
    for epoch in tqdm(range(epochs), desc="Training Epochs", position=0):
        model.train()
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", position=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = correct / total if total > 0 else 0.0
        val_acc = eval(model, val_loader, device)
        accs.append((train_acc, val_acc))
        tqdm.write(f"Epoch [{epoch+1}/{epochs}], Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
    return accs

def make_dataloader(dataset: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    #### データセットの前処理

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = ImageFolder(root=f"{dataset}/train", transform=transform)
    val_dataset = ImageFolder(root=f"{dataset}/val", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Z:\\dataset\\imagenette2-160")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--class_num", type=int, default=10)
    parser.add_argument("--project", type=str, default="project")
    parser.add_argument("--result", type=str, default="result")
    args = parser.parse_args()

    dataset_path: str = args.dataset
    epochs: int = args.epochs
    batch_size: int = args.batch_size
    lr: float = args.lr
    device: torch.device = torch.device(args.device)
    class_num: int = args.class_num
    project: str = args.project
    result: str = args.result

    train_loader, val_loader = make_dataloader(dataset_path, batch_size)

    model = VGG16(output_dim=class_num).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    accs = train(model, train_loader, val_loader, device, epochs, lr)

    # 結果の保存
    os.makedirs(os.path.join(project, result), exist_ok=True)

    # 精度
    result_path = os.path.join(project, result, "accuracy.csv")
    with open(result_path, "w", newline="") as f:
        writer = csv.writer(f)
        for acc in accs:
            writer.writerow([acc])
    print(f"Accuracy results saved to {result_path}")

    # モデル
    model_path = os.path.join(project, result, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 実行時の設定
    config_path = os.path.join(project, result, "config.txt")
    with open(config_path, "w") as f:
        f.write(f"dataset: {dataset_path}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"device: {device}\n")
        f.write(f"class_num: {class_num}\n")
        f.write(f"project: {project}\n")
        f.write(f"result: {result}\n")
    print(f"Configuration saved to {config_path}")

    # 学習曲線の図
    plt.figure()
    plt.plot(range(1, epochs + 1), accs, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    curve_path = os.path.join(project, result, "accuracy_curve.png")
    plt.savefig(curve_path)
    print(f"Accuracy curve saved to {curve_path}")
    plt.close()

    print("Training completed.")

if __name__ == "__main__":
    main()