# google netによるimagenette160, 320の学習
# 呼び出し方
# python main.py 
#   --dataset [dataset, default: Z:\dataset\imagenette2-160]
#   --epochs [number of epochs, default: 100]
#   --batch-size [batch size, default: 32]
#   --lr [learning rate, default: 0.001]
#   --device [device to use, default: cuda if available else cpu]
#   --project [project name, default: project]
#   --result [path to save result, default: result]

import argparse
from tqdm import tqdm
import os

import matplotlib.pyplot as plt

import torch
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from comm.dataset import Dataset, make_dataloader, get_dataset_enum, get_class_num
from GoogleNet.GoogleNet import GoogleNet

def eval(model: torch.nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    correct = 0
    total = 0
    loss_total = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(val_loader, position=1, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            output, aux1, aux2 = model(images)
            loss = criterion(output, labels)

            loss_total += loss.item()
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return loss_total / len(val_loader), correct / total

def train_one_epoch(model: torch.nn.Module, train_loader: DataLoader, optimizer: Optimizer, criterion: nn.Module, device: torch.device) -> tuple[float, float]:

    correct = 0
    total = 0
    loss_total = 0.0
    model.train()
    for images, labels in tqdm(train_loader, position=1, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        output, aux1, aux2 = model(images)

        loss = criterion(output, labels) + 0.3 * criterion(aux1, labels) + 0.3 * criterion(aux2, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        _, predicted = output.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return loss_total / len(train_loader), correct / total

def train(epochs: int, 
          model: torch.nn.Module, 
          train_loader: DataLoader, 
          val_loader: DataLoader, 
          optimizer: Optimizer, 
          criterion: nn.Module, 
          device: torch.device) -> tuple[list[float], list[float], list[float], list[float]]:
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in tqdm(range(epochs), position=0, desc="Epochs"):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval(model, val_loader, criterion, device)

        tqdm.write(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

    return train_loss_history, val_loss_history, train_acc_history, val_acc_history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imagenette2-160")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--project", type=str, default="GoogleNet")
    parser.add_argument("--result", type=str, default="result")
    args = parser.parse_args()

    dataset_enum = get_dataset_enum(args.dataset)
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    device = torch.device(args.device)
    project = args.project
    result = args.result

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor()
    ])

    train_loader, val_loader = make_dataloader(dataset_enum, batch_size, train_transform, val_transform, True)

    model = GoogleNet(classification_num=get_class_num(dataset_enum)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_loss_history, val_loss_history, train_acc_history, val_acc_history = train(
        epochs,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device
    )

    ##### 結果の保存
    os.makedirs(os.path.join(project, result), exist_ok=True)

    # モデルの保存
    torch.save(model.state_dict(), os.path.join(project, result, "model.pth"))

    # 学習結果の保存
    with open(os.path.join(project, result, "result.csv"), "w") as f:
        f.write("train_loss,val_loss,train_acc,val_acc\n")
        for i in range(epochs):
            f.write(f"{train_loss_history[i]},{val_loss_history[i]},{train_acc_history[i]},{val_acc_history[i]}\n")

    # 学習曲線の図の作成
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label="train_loss")
    plt.plot(val_loss_history, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label="train_acc")
    plt.plot(val_acc_history, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(os.path.join(project, result, "learning_curve.png"))
    plt.close()

    print(f"Training complete. Results saved in {os.path.join(project, result)}")

if __name__ == "__main__":
    main()