# NiNの学習スクリプト

import argparse
from tqdm import tqdm
import os
from typing import SupportsFloat

import matplotlib.pyplot as plt

import torch
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from comm.dataset import Dataset, make_dataloader, get_dataset_enum, get_class_num
from comm.filesystem import solve_filename_conflict, solve_foldername_conflict
from comm.train_eval import train_loop, make_log, SchedulerStepTiming
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",          # 評価損失は小さいほどよい
        factor=0.1,          # 学習率を1/10にする
        patience=5,          # 改善しないエポックに5回の猶予
        threshold=0.001,     # 0.001を超える損失低下を改善と判定
        threshold_mode="abs",
        min_lr=1e-6,
    )

    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop((32, 32), scale=(0.5, 1.0)),
    #     transforms.RandomHorizontalFlip(),
    #     # transforms.RandomVerticalFlip(),
    #     transforms.RandomRotation(15),
    #     transforms.ToTensor()
    # ])
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    train_loader, val_loader = make_dataloader(dataset_enum, batch_size=batch_size, train_transform=train_transform, val_transform=val_transform, download=True)

    train_losses, train_accs, val_losses, val_accs, timestamps = train_loop(model, train_loader, val_loader, criterion, optimizer, device, epochs, scheduler_step=lambda m: scheduler.step(m.val_loss), scheduler_step_timing=SchedulerStepTiming.AFTER_EACH_EPOCH)

    # 結果の保存
    make_log(os.path.join(project, result), model, train_losses, train_accs, val_losses, val_accs, timestamps, args)

if __name__ == '__main__':
    main()
