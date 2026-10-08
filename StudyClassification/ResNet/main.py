# ResNetによるimagenette160, 320の学習
# 呼び出し方
# python main.py 
#   --dataset [dataset, default: imagenette2-160]
#   --model [model path, default: None]
#   --epochs [number of epochs, default: 100]
#   --batch-size [batch size, default: 32]
#   --lr [learning rate, default: 0.0001]
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
from comm.filesystem import solve_filename_conflict, solve_foldername_conflict
from comm.train_eval import train_loop, make_log, log_args
from ResNet.network import ResNet18, ResNet50, PreactivationResNet18

def make_model(model_name: str, use_pre_activation: bool, user_model_path: str, class_num: int, device: torch.device) -> nn.Module:
    if user_model_path is not None:
        model = torch.load(user_model_path).to(device)
    else:
        if use_pre_activation:
            if model_name == "ResNet18":
                model = PreactivationResNet18(num_classes=class_num).to(device)
            else:
                raise ValueError(f"Unsupported pre-activation model: {model_name}")
        else:
            if model_name == "ResNet18":
                model = ResNet18(num_classes=class_num).to(device)
            elif model_name == "ResNet50":
                model = ResNet50(num_classes=class_num).to(device)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imagenette2-160')
    parser.add_argument('--user-model', type=str, help='Path to a custom user-defined model', default=None)
    parser.add_argument('--pre-activation', action='store_true', help='Use pre-activation ResNet')
    parser.add_argument('--model', type=str, default="ResNet18")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--project', type=str, default='ResNet')
    parser.add_argument('--result', type=str, default='result')
    args = parser.parse_args()

    dataset_enum = get_dataset_enum(args.dataset)
    model_path = args.user_model
    use_pre_activation = args.pre_activation
    model_name = args.model
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    device = torch.device(args.device)
    project = args.project
    result = args.result
    class_num = get_class_num(dataset_enum)

    model = make_model(model_name, use_pre_activation, model_path, class_num, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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

    train_loader, val_loader = make_dataloader(dataset_enum, batch_size=batch_size, train_transform=train_transform, val_transform=val_transform, download=True)

    # 学習の実行
    train_losses, train_accs, val_losses, val_accs, timestamps = train_loop(model, train_loader, val_loader, criterion, optimizer, device, epochs)

    # ログの保存
    make_log(os.path.join(project, result), model, train_losses, train_accs, val_losses, val_accs, timestamps)
    log_args(os.path.join(project, result), args)

if __name__ == "__main__":
    main()