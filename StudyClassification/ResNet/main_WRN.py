# Wide ResNetの学習スクリプト

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
from comm.train_eval import train_loop, make_log
from ResNet.network import WideResNet18

def make_model(model_name: str, model_path: str, widen_factor: int, drop_rate: float, class_num: int, pre_activation: bool, device: torch.device) -> nn.Module:
    if model_path is not None:
        model = torch.load(model_path).to(device)
    else:
        model = WideResNet18(class_num, widen_factor, drop_rate, pre_activation).to(device)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imagenette2-160')
    parser.add_argument('--user-model', type=str, help='Path to a custom user-defined model', default=None)
    parser.add_argument('--pre-activation', action='store_true', help='Use pre-activation ResNet')
    parser.add_argument('--model', type=str, default="WideResNet")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--widen-factor', type=float, required=True, help='Widen factor of the Wide ResNet')
    parser.add_argument('--drop-rate', type=float, default=0.0, help='Dropout rate of the Wide ResNet')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--project', type=str, default='run')
    parser.add_argument('--result', type=str, default='WideResNet')
    args = parser.parse_args()

    dataset_enum = get_dataset_enum(args.dataset)
    model_path = args.user_model
    use_pre_activation = args.pre_activation
    model_name = args.model
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    widen_factor = args.widen_factor
    drop_rate = args.drop_rate
    device = torch.device(args.device)
    project = args.project
    result = args.result
    class_num = get_class_num(dataset_enum)

    model = make_model(model_name, model_path, widen_factor, drop_rate, class_num, use_pre_activation, device)

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

    # 結果の保存
    make_log(os.path.join(project, result), model, train_losses, train_accs, val_losses, val_accs, timestamps, args)
    
if __name__ == '__main__':
    main()