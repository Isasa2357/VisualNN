
# MobileNet v1の学習スクリプト

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
from comm.train_eval import train_loop
from comm.train_eval import make_log

from MobileNet_v1.network import MobileNetV1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imagenette2-160')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--project', type=str, default='MobileNet_v1_results')
    parser.add_argument('--result', type=str, default='result')
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

    if model_path == None:
        model = MobileNetV1(class_num)
    else:
        model = MobileNetV1(class_num)
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accs, val_losses, val_accs, timestamp_history = train_loop(model, train_loader, val_loader, criterion, optimizer, device, epochs)
    
    save_root = solve_foldername_conflict(os.path.join(project, result))
    make_log(save_root, model, train_losses, train_accs, val_losses, val_accs, timestamp_history, args)


if __name__ == "__main__":
    main()