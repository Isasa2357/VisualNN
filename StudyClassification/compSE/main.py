# google netによるimagenette160, 320の学習

import argparse
from tqdm import tqdm
import os
from enum import Enum

import matplotlib.pyplot as plt

import torch
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from comm.dataset import Dataset, make_dataloader, get_dataset_enum, get_class_num
from compSE.nn import NN
from comm.train_eval import train_loop, TrainLoop, EpochMetrics, SchedulerStepTiming, EarlyStopSettings, make_log, default_correct_calculation, default_loss_calculation, default_correct_fn, default_loss_fn

def parse_args():
    parser = argparse.ArgumentParser(description="Xception Training")

    # training settings
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dataset", type=str, default="imagenet100", help="Dataset name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision for training")
    parser.add_argument("--early-stop-patience", type=int, default=30, help="Patience for early stopping")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001, help="Minimum change to qualify as improvement for early stopping")
    parser.add_argument("--early-stop-warmup-epochs", type=int, default=30, help="Number of warmup epochs before early stopping starts checking")
    parser.add_argument("--dataloader-workers", type=int, default=6)
    parser.add_argument("--dataloader-prefetch", type=int, default=2)
    parser.add_argument("--label-smooth", type=float, default=0.0)

    # model setting
    parser.add_argument("--use-se", action="store_true",help="SEブロックを適用するかどうか")

    # logging settings
    parser.add_argument('--dir', type=str, default='./', help='Directory for the project')
    parser.add_argument('--project', type=str, default='run')
    parser.add_argument('--result', type=str, default='compSE')

    # other settings
    parser.add_argument('--no-robocopy', action='store_true', help='Use robocopy to copy the dataset')

    return parser.parse_args()

def build_model(class_num: int, se: bool):
    return NN(class_num, se)

def main():
    args = parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    dataset_enum = get_dataset_enum(args.dataset)
    class_num = get_class_num(dataset_enum)
    device = torch.device(args.device)
    use_amp = args.use_amp
    early_stop_patience = args.early_stop_patience
    early_stop_min_delta = args.early_stop_min_delta
    early_stop_warmup_epochs = args.early_stop_warmup_epochs
    dataloader_workers = args.dataloader_workers
    dataloader_prefetch = args.dataloader_prefetch
    label_smooth = args.label_smooth

    use_se = args.use_se
    model = build_model(class_num, use_se).to(device)

    dir = args.dir
    project = args.project
    result = args.result

    no_robocopy = args.no_robocopy

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=2, 
        gamma=0.94
    )

    resize_size = (224, 224)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(resize_size, (0.3, 1)), 
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(15), 
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor()
    ])

    train_loader, val_loader = make_dataloader(
        dataset_enum, 
        batch_size, 
        train_transform=train_transform, 
        val_transform=val_transform, 
        download=True, 
        robocopy=not no_robocopy,
        train_drop_last=True,
        val_drop_last=False, 
        num_workers=dataloader_workers, 
        prefetch_factor=dataloader_prefetch
    )

    # lrの更新関数を定義
    def scheduler_step(epoch_metrics: EpochMetrics):
        scheduler.step()

    train_loop_instance = TrainLoop(
        model=model, 
        num_epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device, 
        log_dir=os.path.join(dir, project, result), 
        use_amp=use_amp
    )
    
    train_loop_instance.set_early_stopping(EarlyStopSettings(
        patience=early_stop_patience,
        min_delta=early_stop_min_delta, 
        warmup_epochs=early_stop_warmup_epochs
    ))

    train_loop_instance.set_lr_scheduler(scheduler_step)

    train_loop_instance.start()

    train_loop_instance.make_log(args)

if __name__ == "__main__":
    main()
