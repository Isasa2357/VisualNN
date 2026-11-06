
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
from comm.train_eval import train_loop, make_log, EpochMetrics, SchedulerStepTiming, EarlyStopSettings
from Xception.nn import Xception

def parse_args():
    parser = argparse.ArgumentParser(description="Xception Training")

    # training settings
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--dataset", type=str, default="imagenette2-320", help="Dataset name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision for training")
    parser.add_argument("--early_stop_patience", type=int, default=30, help="Patience for early stopping")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.001, help="Minimum change to qualify as improvement for early stopping")
    parser.add_argument("--early_stop_warmup_epochs", type=int, default=30, help="Number of warmup epochs before early stopping starts checking")

    # model setting
    parser.add_argument("--model", type=str, default=None, help="Model path")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for the classifier")

    # logging settings
    parser.add_argument('--dir', type=str, default='./', help='Directory for the project')
    parser.add_argument('--project', type=str, default='run')
    parser.add_argument('--result', type=str, default='Xception')

    # other settings
    parser.add_argument('--no-robocopy', action='store_true', help='Use robocopy to copy the dataset')

    return parser.parse_args()

def main():

    args = parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    dataset_enum = get_dataset_enum(args.dataset)
    num_classes = get_class_num(dataset_enum)
    device = torch.device(args.device)
    use_amp = args.use_amp
    early_stop_patience = args.early_stop_patience
    early_stop_min_delta = args.early_stop_min_delta
    early_stop_warmup_epochs = args.early_stop_warmup_epochs
    model_path = args.model
    dropout = args.dropout
    dir = args.dir
    project = args.project
    result = args.result
    no_robocopy = args.no_robocopy

    # モデルの作成
    if model_path is None:
        model = Xception(num_classes=num_classes, dropout=dropout).to(device)
    else:
        model = torch.load(model_path).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=2, 
        gamma=0.94
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((299, 299), (0.3, 1)), 
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(15), 
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])

    train_loader, val_loader = make_dataloader(
        dataset_enum, 
        batch_size, 
        train_transform=train_transform, 
        val_transform=val_transform, 
        download=True, 
        robocopy=not no_robocopy
    )

    # lrの更新関数を定義
    def scheduler_step(epoch_metrics: EpochMetrics):
        scheduler.step()

    # 学習の実行
    train_losses, train_accs, val_losses, val_accs, timestamps = \
        train_loop(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer, 
            device, 
            epochs, 
            scheduler_step=scheduler_step, 
            scheduler_step_timing=SchedulerStepTiming.AFTER_EACH_EPOCH,
            use_amp=use_amp,
            early_stop_settings=EarlyStopSettings(
                patience=early_stop_patience,
                min_delta=early_stop_min_delta, 
                warmup_epochs=early_stop_warmup_epochs
            )
        )

    # 結果の保存
    make_log(
        os.path.join(dir, project, result),
        model, 
        train_losses, 
        train_accs,
        val_losses,
        val_accs,
        timestamps, 
        args
    )

if __name__ == "__main__":
    main()