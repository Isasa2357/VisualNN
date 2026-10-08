import argparse
from tqdm import tqdm
import os
import csv

import matplotlib.pyplot as plt

import torch
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from comm.filesystem import solve_foldername_conflict
from comm.dataset import Dataset, make_dataloader, get_dataset_enum, get_class_num
from DenseNet.network import DenseNet121

def eval(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, position=1, leave=False, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(val_loader), correct / total

def train_one_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: Optimizer, device: torch.device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    for inputs, targets in tqdm(train_loader, position=1, leave=False, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total += targets.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(train_loader), correct / total

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
          criterion: nn.Module, optimizer: Optimizer, device: torch.device, num_epochs: int) -> tuple[list[float], list[float], list[float], list[float]]:
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in tqdm(range(num_epochs), position=0, leave=True, desc="Epochs"):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return train_losses, train_accs, val_losses, val_accs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imagenette2-160')
    parser.add_argument('--model', type=str, default="ResNet18")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--project', type=str, default='project')
    parser.add_argument('--result', type=str, default='result')
    parser.add_argument('--growth-rate', type=int, default=32)
    args = parser.parse_args()

    dataset_enum = get_dataset_enum(args.dataset)
    model_path = args.model
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    device = torch.device(args.device)
    project = args.project
    result = args.result
    class_num = get_class_num(dataset_enum)
    growth_rate = args.growth_rate

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
    model = DenseNet121(class_num, growth_rate=growth_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accs, val_losses, val_accs = train(model, train_loader, val_loader, criterion, optimizer, device, epochs)

    ##### 結果の保存

    result_dir = solve_foldername_conflict(os.path.join(project, result))
    os.makedirs(result_dir, exist_ok=True)

    # 推移をcsvで保存
    csv_path = os.path.join(result_dir, "training_log.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for epoch in range(epochs):
            writer.writerow([epoch+1, train_losses[epoch], train_accs[epoch], val_losses[epoch], val_accs[epoch]])

    # モデルを保存
    model_path = os.path.join(result_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    # 学習曲線をpltで保存

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_accs, label='Train Acc')
    plt.plot(range(1, epochs+1), val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt_path = os.path.join(result_dir, "training_curve.png")
    plt.savefig(plt_path)
    plt.close()

    print(f"Results saved in {result_dir}")

if __name__ == "__main__":
    main()