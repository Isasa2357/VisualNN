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
from Inception.nn import InceptionV1, InceptionV2, InceptionV3, InceptionV4, InceptionResNetV1, InceptionResNetV2
from comm.train_eval import train_loop, TrainLoop, EpochMetrics, SchedulerStepTiming, EarlyStopSettings, make_log, default_correct_calculation, default_loss_calculation, default_correct_fn, default_loss_fn

class InceptionArchitecture(Enum):
    V1=0
    V2=1
    V3=2
    V4=3
    ResV1=4
    ResV2=5

def conv_modelV_to_archtectureEnum(model_v: str) -> InceptionArchitecture:
    if model_v == "1":
        return InceptionArchitecture.V1
    elif model_v == "2":
        return InceptionArchitecture.V2    
    elif model_v == "3":
        return InceptionArchitecture.V3
    elif model_v == "4":
        return InceptionArchitecture.V4
    elif model_v == "resv1":
        return InceptionArchitecture.ResV1
    elif model_v == "resv2":
        return InceptionArchitecture.ResV2
    else:
        raise ValueError(f"Unsupported model version: {model_v}")

def parse_args():
    parser = argparse.ArgumentParser(description="Xception Training")

    # training settings
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dataset", type=str, default="imagenette2-320", help="Dataset name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision for training")
    parser.add_argument("--early_stop_patience", type=int, default=30, help="Patience for early stopping")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.001, help="Minimum change to qualify as improvement for early stopping")
    parser.add_argument("--early_stop_warmup_epochs", type=int, default=30, help="Number of warmup epochs before early stopping starts checking")

    # model setting
    parser.add_argument("--model", type=str, default=None, help="Model path")
    parser.add_argument("--model_v", type=str, required=True, help="Model version (1, 2, 3, 4, resv1, resv2)")

    # logging settings
    parser.add_argument('--dir', type=str, default='./', help='Directory for the project')
    parser.add_argument('--project', type=str, default='run')
    parser.add_argument('--result', type=str, default='Inception')

    # other settings
    parser.add_argument('--no-robocopy', action='store_true', help='Use robocopy to copy the dataset')

    return parser.parse_args()

def build_model(model_name: str|None, archtecture: InceptionArchitecture, class_num: int):

    model: nn.Module
    if archtecture == InceptionArchitecture.V1:
        model = InceptionV1(class_num)
    elif archtecture == InceptionArchitecture.V2:
        model = InceptionV2(class_num)
    elif archtecture == InceptionArchitecture.V3:
        model = InceptionV3(class_num)
    elif archtecture == InceptionArchitecture.V4:
        model = InceptionV4(class_num)
    elif archtecture == InceptionArchitecture.ResV1:
        model = InceptionResNetV1(class_num)
    elif archtecture == InceptionArchitecture.ResV2:
        model = InceptionResNetV2(class_num)
    else:
        raise ValueError(f"Unsupported model version: {model_v}")

    if model_name is not None and os.path.exists(model_name):
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
    archtecture = conv_modelV_to_archtectureEnum(args.model_v)

    dir = args.dir
    project = args.project
    result = args.result

    no_robocopy = args.no_robocopy

    # モデルの作成
    model = build_model(model_name, archtecture, class_num).to(device)

    criterion = nn.CrossEntropyLoss() if archtecture in [InceptionArchitecture.V3, InceptionArchitecture.V4, InceptionArchitecture.ResV1, InceptionArchitecture.ResV2] else nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=2, 
        gamma=0.94
    )

    resize_size = (299, 299) if archtecture == InceptionArchitecture.V1 else (224, 224)

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

    def correct_fn_v1_3_top1_5(out: torch.Tensor, target: torch.Tensor) -> tuple[int, int, int]:
        out = out[0]
        return default_correct_fn(out, target)

    loss_fn = \
        train_loss_fn_v1 if archtecture == InceptionArchitecture.V1 \
        else train_loss_fn_v3 if archtecture == InceptionArchitecture.V3 \
        else default_loss_fn

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

    train_loop_instance.set_loss_fn(loss_fn)

    if archtecture in [InceptionArchitecture.V1, InceptionArchitecture.V3]:
        train_loop_instance.set_correct_fn(correct_fn_v1_3_top1_5)
    
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