# NiNの学習スクリプト

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
from NiN.nn import NiN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--user-model', type=str, help='Path to a custom user-defined model', default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--project', type=str, default='run')
    parser.add_argument('--result', type=str, default='NiN')
    args = parser.parse_args()

    dataset_enum = get_dataset_enum(args.dataset)
    model_path = args.user_model
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    dropout = args.dropout
    device = torch.device(args.device)
    project = args.project
    result = args.result
    class_num = get_class_num(dataset_enum)

    model: nn.Module
    if model_path is not None:
        model = NiN(class_num=class_num, dropout=dropout)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
    else:
        model = NiN(class_num=class_num, dropout=dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop((32, 32), scale=(0.5, 1.0)),
    #     transforms.RandomHorizontalFlip(),
    #     # transforms.RandomVerticalFlip(),
    #     transforms.RandomRotation(15),
    #     transforms.ToTensor()
    # ])
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor()
    ])

    train_loader, val_loader = make_dataloader(dataset_enum, batch_size=batch_size, train_transform=train_transform, val_transform=val_transform, download=True)

    train_losses, train_accs, val_losses, val_accs, timestamps = train_loop(model, train_loader, val_loader, criterion, optimizer, device, epochs)

    # 結果の保存
    make_log(os.path.join(project, result), model, train_losses, train_accs, val_losses, val_accs, timestamps)
    log_args(os.path.join(project, result), args)

if __name__ == '__main__':
    main()
