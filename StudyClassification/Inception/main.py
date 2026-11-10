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
from Inception.nn import InceptionV1, InceptionV2, InceptionV3
from comm.train_eval import train_loop, EpochMetrics, SchedulerStepTiming, EarlyStopSettings, make_log, default_correct_calculation, default_loss_calculation

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
    parser.add_argument("--model", type=str, required=True, help="Model path or name")
    parser.add_argument("--model_v", type=int, required=True, help="Model version (1, 2, or 3)")

    # logging settings
    parser.add_argument('--dir', type=str, default='./', help='Directory for the project')
    parser.add_argument('--project', type=str, default='run')
    parser.add_argument('--result', type=str, default='Inception')

    # other settings
    parser.add_argument('--no-robocopy', action='store_true', help='Use robocopy to copy the dataset')

    return parser.parse_args()

def build_model(model_name: str, model_v: int, class_num: int):

    model: nn.Module
    if model_v == 1:
        model = InceptionV1(class_num)
    elif model_v == 2:
        model = InceptionV2(class_num)
    elif model_v == 3:
        model = InceptionV3(class_num)
    else:
        raise ValueError(f"Unsupported model version: {model_v}")

    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name))

    return model

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

    model_name = args.model
    model_v = args.model_v

    dir = args.dir
    project = args.project
    result = args.result

    no_robocopy = args.no_robocopy

    # モデルの作成
    model = build_model(model_name, model_v, class_num).to(device)

    criterion = nn.CrossEntropyLoss() if model_v != 3 else nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=2, 
        gamma=0.94
    )

    resize_size = (299, 299) if model_v != 1 else (224, 224)

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
        val_drop_last=False
    )

    # lrの更新関数を定義
    def scheduler_step(epoch_metrics: EpochMetrics):
        scheduler.step()

    def train_loss_fn_v1(out: torch.Tensor, target: torch.Tensor, criterion: nn.Module, mode: str):
        if mode == "train":
            return criterion(out[0], target) + 0.3 * criterion(out[1], target) + 0.3 * criterion(out[2], target)
        else:
            return criterion(out[0], target)

    def train_loss_fn_v3(out: torch.Tensor, target: torch.Tensor, criterion: nn.Module, mode: str):
        if mode == "train":
            return criterion(out[0], target) + 0.4 * criterion(out[1], target)
        else:
            return nn.CrossEntropyLoss()(out[0], target)

    def correct_fn_v1_3(out: torch.Tensor, target: torch.Tensor) -> tuple[int, int]:
        main_out = out[0]
        _, predicted = torch.max(main_out, 1)
        correct = int((predicted == target).sum().item())
        total = target.size(0)
        return correct, total

    loss_fn = \
        train_loss_fn_v1 if model_v == 1 \
        else default_loss_calculation if model_v == 2 \
        else train_loss_fn_v3
    
    correct_fn = correct_fn_v1_3 if model_v in [1, 3] else default_correct_calculation

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
            loss_calculation_fn=loss_fn, 
            correct_calculation_fn=correct_fn,
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