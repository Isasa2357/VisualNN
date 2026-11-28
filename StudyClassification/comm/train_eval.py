# 学習と評価ループの汎用型

import os
import sys
import csv
import argparse

from collections.abc import Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from comm.filesystem import solve_foldername_conflict

class SchedulerStepTiming(Enum):
    AFTER_EACH_EPOCH = 1
    AFTER_EACH_BATCH = 2

@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float

SchedulerStep = Callable[[EpochMetrics], None]

@dataclass(frozen=True)
class EarlyStopSettings:
    patience: int
    min_delta: float
    warmup_epochs: int

@dataclass(frozen=True)
class EarlyStopState:
    early_stop_settings: EarlyStopSettings
    best_val_loss: float=sys.float_info.max
    counter: int=0

def default_loss_calculation(out: torch.Tensor, target: torch.Tensor, criterion: nn.Module, mode: str):
    '''
    デフォルトの損失計算関数
    '''
    return criterion(out, target)

def default_correct_calculation(out: torch.Tensor, target: torch.Tensor) -> tuple[int, int]:
    '''
    デフォルトの正解数計算関数

    args:
        out: モデルの出力テンソル
        target: 正解ラベルのテンソル

    returns:
        正解数とサンプル数のタプル (correct, total)
    '''
    _, predicted = torch.max(out, 1)
    correct = int((predicted == target).sum().item())
    total = target.size(0)
    return correct, total

def eval(
        model: nn.Module, 
        data_loader: DataLoader, 
        criterion: nn.Module, 
        device: torch.device, 
        loss_calculation_fn: Callable[[torch.Tensor, torch.Tensor, nn.Module, str], torch.Tensor] = default_loss_calculation, 
        correct_calculation_fn: Callable[[torch.Tensor, torch.Tensor], tuple[int, int]] = default_correct_calculation
    ) -> tuple[float, float, datetime, datetime]:

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    param_dtype = next(model.parameters()).dtype

    start = datetime.now()

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating", leave=False, position=1):
            inputs, targets = inputs.to(device, non_blocking=True, dtype=param_dtype), targets.to(device, non_blocking=True)
            outputs = model(inputs)

            loss = loss_calculation_fn(outputs, targets, criterion, "eval")
            total_loss += loss.item() * inputs.size(0)

            correct, total = correct_calculation_fn(outputs, targets)
            total_correct += correct
            total_samples += total

    end = datetime.now()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy, start, end

def check_weights(model):
    return [
        name
        for name, p in model.named_parameters()
        if not torch.isfinite(p).all().item()
    ]

def train_one_epoch(
        epoch: int,
        model: nn.Module, 
        data_loader: DataLoader, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer, 
        device: torch.device, 
        loss_calculation_fn: Callable[[torch.Tensor, torch.Tensor, nn.Module, str], torch.Tensor]=default_loss_calculation, 
        correct_calculation_fn: Callable[[torch.Tensor, torch.Tensor], tuple[int, int]]=default_correct_calculation,
        scheduler_step: SchedulerStep | None = None,
        scaler: torch.amp.GradScaler | None = None,
    ) -> tuple[float, float, datetime, datetime]:

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    param_dtype = next(model.parameters()).dtype

    start = datetime.now()

    for inputs, targets in tqdm(data_loader, desc="Training", leave=False, position=1):
        inputs, targets = inputs.to(device, non_blocking=True, dtype=param_dtype), targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        if scaler is None:
            outputs = model(inputs)
            loss = loss_calculation_fn(outputs, targets, criterion, "train")
            loss.backward()
            optimizer.step()
        else:
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = loss_calculation_fn(outputs, targets, criterion, "train")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * inputs.size(0)

        correct, total = correct_calculation_fn(outputs, targets)
        total_correct += correct
        total_samples += total

        if scheduler_step is not None:
            metrics = EpochMetrics(
                epoch=epoch, 
                train_loss=total_loss / total_samples,
                train_acc=total_correct / total_samples, 
                val_loss=0.0, 
                val_acc=0.0,
            )
            before_scheduler_lr = optimizer.param_groups[0]['lr']
            scheduler_step(metrics)
            after_scheduler_lr = optimizer.param_groups[0]['lr']
            tqdm.write(f"Scheduler step executed at epoch {epoch+1}, lr changed from {before_scheduler_lr} to {after_scheduler_lr}")

    end = datetime.now()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy, start, end

def train_loop(
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer, 
        device: torch.device, 
        num_epochs: int,
        loss_calculation_fn: Callable[[torch.Tensor, torch.Tensor, nn.Module, str], torch.Tensor] = default_loss_calculation, 
        correct_calculation_fn: Callable[[torch.Tensor, torch.Tensor], tuple[int, int]] = default_correct_calculation, 
        scheduler_step: SchedulerStep | None = None,
        scheduler_step_timing: SchedulerStepTiming | None = None,
        use_amp: bool = False,
        early_stop_settings: EarlyStopSettings | None = None,
    ) -> tuple[list[float], list[float], list[float], list[float], list[tuple[datetime, datetime, datetime, datetime]]]:

    scaler: torch.amp.GradScaler | None = torch.amp.GradScaler() if use_amp else None

    early_stop_state: EarlyStopState | None = EarlyStopState(
        early_stop_settings=early_stop_settings,
        best_val_loss=float('inf'),
        counter=0
    ) if early_stop_settings is not None else None

    train_loss_history: list[float] = list()
    train_accs_history: list[float] = list()
    val_loss_history: list[float] = list()
    val_accs_history: list[float] = list()
    timestamp_history: list[tuple[datetime, datetime, datetime, datetime]] = list()

    for epoch in tqdm(range(num_epochs), desc="Training Epochs", leave=True, position=0):
        train_loss, train_acc, train_start, train_end = \
            train_one_epoch(
                epoch, 
                model, 
                train_loader, 
                criterion, 
                optimizer, 
                device, 
                loss_calculation_fn, 
                correct_calculation_fn,
                scheduler_step if scheduler_step_timing == SchedulerStepTiming.AFTER_EACH_BATCH else None, 
                scaler=scaler
            )
        val_loss, val_acc, val_start, val_end = \
            eval(
                model, 
                val_loader, 
                criterion, 
                device, 
                loss_calculation_fn,
                correct_calculation_fn,
            )

        train_duration: str = str(train_end - train_start).split('.')[0]
        val_duration: str = str(val_end - val_start).split('.')[0]
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, duration: {train_duration} - "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, duration: {val_duration}")

        train_loss_history.append(train_loss)
        train_accs_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_accs_history.append(val_acc)
        timestamp_history.append((train_start, train_end, val_start, val_end))

        # early stop処理
        if early_stop_state is not None and epoch >= early_stop_state.early_stop_settings.warmup_epochs:

            if val_loss < early_stop_state.best_val_loss - early_stop_state.early_stop_settings.min_delta:
                # 改善した場合
                early_stop_state = EarlyStopState(
                    early_stop_settings=early_stop_state.early_stop_settings,
                    best_val_loss=val_loss,
                    counter=0
                )
            else:
                # 改善しなかった場合
                early_stop_state = EarlyStopState(
                    early_stop_settings=early_stop_state.early_stop_settings,
                    best_val_loss=early_stop_state.best_val_loss,
                    counter=early_stop_state.counter + 1
                )

            # early stop判定
            if early_stop_state.counter >= early_stop_state.early_stop_settings.patience:
                tqdm.write(f"Early stopping at epoch {epoch+1}")
                break

        if scheduler_step is not None and scheduler_step_timing == SchedulerStepTiming.AFTER_EACH_EPOCH:
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )

            before_scheduler_lr = optimizer.param_groups[0]['lr']
            scheduler_step(metrics)
            after_scheduler_lr = optimizer.param_groups[0]['lr']

            if before_scheduler_lr != after_scheduler_lr:
                tqdm.write(f"Scheduler step executed at epoch {epoch+1}, lr changed from {before_scheduler_lr} to {after_scheduler_lr}")    

    return train_loss_history, train_accs_history, val_loss_history, val_accs_history, timestamp_history

def make_log(root: str, model: nn.Module, train_loss_history: list[float], train_accs_history: list[float], val_loss_history: list[float], val_accs_history: list[float], timestamp_history: list[tuple[datetime, datetime, datetime, datetime]], args: argparse.Namespace) -> None:
    root = solve_foldername_conflict(root)
    
    # フォルダ作成
    os.makedirs(root, exist_ok=True)

    # モデルの保存
    model_path = os.path.join(root, "model.pth")
    torch.save(model.state_dict(), model_path)

    # 学習曲線のプロット
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs_history, label="Train Acc")
    plt.plot(val_accs_history, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt_path = os.path.join(root, "training_curve.png")
    plt.savefig(plt_path)
    plt.close()

    # 学習結果の推移の保存
    history_filename = os.path.join(root, "history.csv")
    with open(history_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train duration", "val duration", "train acc", "train loss", "val acc", "val loss"])
        for i in range(len(train_loss_history)):
            train_duration = timestamp_history[i][1] - timestamp_history[i][0]
            val_duration = timestamp_history[i][3] - timestamp_history[i][2]
            writer.writerow([
                i + 1,
                f"{train_duration}",
                f"{val_duration}",
                train_accs_history[i],
                train_loss_history[i],
                val_accs_history[i],
                val_loss_history[i]
            ])

    # 引数の保存
    args_filename = os.path.join(root, "args.txt")
    with open(args_filename, mode="w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    print(f"Training history saved to {history_filename}")

def default_loss_fn(outputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module, phase: str) -> torch.Tensor:
    return criterion(outputs, targets)

def default_correct_fn(outputs: torch.Tensor, targets: torch.Tensor) -> tuple[int, int, int]:
    '''
    正答個数を計算する

    args:
        outputs: モデルの出力テンソル
        targets: 正解ラベルのテンソル
    
    returns:
        tuple[int, int, int]: top1正答数, top5正答数, サンプル数
    '''
    # top1
    _, pred = torch.max(outputs, dim=1)
    top1_correct = (pred == targets).sum().item()

    # top5
    top5_id = outputs.topk(5, dim=1).indices
    top5_correct = (top5_id == targets.unsqueeze(1)).any(dim=1).sum().item()

    return int(top1_correct), int(top5_correct), targets.size(0)

class TrainLoop:
    def __init__(
            self, 
            model: nn.Module, 
            num_epochs: int, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module, 
            optimizer: torch.optim.Optimizer,
            device: torch.device, 
            log_dir: str,
            use_amp: bool = False
    ):
        self._model = model
        self._best_model_state: dict[str, torch.Tensor] = {}
        self._best_model_acc: float = 0.0
        self._num_epochs = num_epochs
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._criterion = criterion
        self._optimizer = optimizer
        self._device = device
        self._use_amp = use_amp
        self._scaler = torch.amp.GradScaler()

        # ロス計算式
        self._loss_fn: Callable[[torch.Tensor, torch.Tensor, nn.Module, str], torch.Tensor] = default_loss_fn

        # 正答率計算式
        self._correct_fn: Callable[[torch.Tensor, torch.Tensor], tuple[int, int, int]] = default_correct_fn

        # lrスケジューラ
        self._use_lr_scheduler = False
        self._lr_scheduler: SchedulerStep=lambda metrics: None

        # early stopping
        self._use_early_stopping = False
        self._early_stop_state: EarlyStopState=EarlyStopState(
            EarlyStopSettings(patience=0, min_delta=0.0, warmup_epochs=0), 
            sys.float_info.max, 
            0
        )

        # ログ出力先
        self._log_dir = solve_foldername_conflict(log_dir)

        # 学習中のデータ
        self._train_loss_history: list[float] = []
        self._train_top1_acc_history: list[float] = []
        self._train_top5_acc_history: list[float] = []
        self._train_start_history: list[datetime] = []
        self._train_end_history: list[datetime] = []
        self._val_loss_history: list[float] = []
        self._val_top1_acc_history: list[float] = []
        self._val_top5_acc_history: list[float] = []
        self._val_start_history: list[datetime] = []
        self._val_end_history: list[datetime] = []

    def set_loss_fn(self, loss_fn: Callable[[torch.Tensor, torch.Tensor, nn.Module, str], torch.Tensor]):
        self._loss_fn = loss_fn

    def set_correct_fn(self, correct_fn: Callable[[torch.Tensor, torch.Tensor], tuple[int, int, int]]):
        self._correct_fn = correct_fn

    def set_lr_scheduler(self, lr_scheduler: SchedulerStep):
        self._lr_scheduler = lr_scheduler
        self._use_lr_scheduler = True

    def set_early_stopping(self, early_stop_setting: EarlyStopSettings):
        self._use_early_stopping = True
        self._early_stop_state = EarlyStopState(
            early_stop_setting, 
            sys.float_info.max, 
            0
        )

    def _train_one_epoch(self) -> tuple[float, float, float, datetime, datetime]:
        '''
        1エポック分の学習を行う
        '''

        self._model.train()
        total_loss = 0.0
        total_top5_correct = 0
        total_top1_correct = 0
        total_samples = 0
    
        start = datetime.now()

        for inputs, targets in tqdm(self._train_loader, desc="Training", leave=False, position=1):
            # デバイスにデータを転送
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            self._optimizer.zero_grad()

            # 学習
            if self._use_amp:
                with torch.amp.autocast(device_type=self._device.type, dtype=torch.bfloat16):
                    outputs = self._model(inputs)
                    loss = self._loss_fn(outputs, targets, self._criterion, "train")
                self._scaler.scale(loss).backward()
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                outputs = self._model(inputs)
                loss = self._loss_fn(outputs, targets, self._criterion, "train")
                loss.backward()
                self._optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            top1_correct, top5_correct, total = self._correct_fn(outputs, targets)

            total_top1_correct += top1_correct
            total_top5_correct += top5_correct
            total_samples += total

        end = datetime.now()

        avg_loss = total_loss / total_samples
        top1_acc = total_top1_correct / total_samples
        top5_acc = total_top5_correct / total_samples

        return avg_loss, top1_acc, top5_acc, start, end

    def _eval(self) -> tuple[float, float, float, datetime, datetime]:
        '''
        評価を行う
        '''
        self._model.eval()

        total_loss = 0.0
        total_top1_correct = 0
        total_top5_correct = 0
        total_samples = 0

        start = datetime.now()

        with torch.no_grad():
            for inputs, targets in tqdm(self._val_loader, desc="Evaluating", leave=False, position=1):
                # デバイスにデータを転送
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # 推論
                outputs = self._model(inputs)

                # ロスを計算
                loss = self._loss_fn(outputs, targets, self._criterion, "val")
                total_loss += loss.item() * inputs.size(0)

                # 正答数を計算
                top1_correct, top5_correct, total = self._correct_fn(outputs, targets)
                total_top1_correct += top1_correct
                total_top5_correct += top5_correct
                total_samples += total

        end = datetime.now()

        avg_loss = total_loss / total_samples
        top1_acc = total_top1_correct / total_samples
        top5_acc = total_top5_correct / total_samples

        return avg_loss, top1_acc, top5_acc, start, end

    def start(self):
        '''
        学習を行う
        '''

        for epoch in tqdm(range(self._num_epochs), desc="Epochs", leave=True, position=0):

            # training
            train_loss, train_top1_acc, train_top5_acc, train_start, train_end = self._train_one_epoch()

            # evaluation
            val_loss, val_top1_acc, val_top5_acc, val_start, val_end = self._eval()

            # 集計
            self._train_start_history.append(train_start)
            self._train_end_history.append(train_end)
            self._val_start_history.append(val_start)
            self._val_end_history.append(val_end)

            self._train_loss_history.append(train_loss)
            self._train_top1_acc_history.append(train_top1_acc)
            self._train_top5_acc_history.append(train_top5_acc)
            self._val_loss_history.append(val_loss)
            self._val_top1_acc_history.append(val_top1_acc)
            self._val_top5_acc_history.append(val_top5_acc)

            # best modelの更新
            if val_top1_acc > self._best_model_acc:
                self._best_model_acc = val_top1_acc
                self._best_model_state = self._model.state_dict()
            
            # early stopping
            if self._use_early_stopping:
                ### early stop stateの更新

                if val_loss < self._early_stop_state.best_val_loss - self._early_stop_state.early_stop_settings.min_delta:
                    # 改善した場合
                    self._early_stop_state = EarlyStopState(
                        early_stop_settings=self._early_stop_state.early_stop_settings,
                        best_val_loss=val_loss, 
                        counter=0
                    )
                else:
                    # 改善しなかった場合
                    self._early_stop_state = EarlyStopState(
                        early_stop_settings=self._early_stop_state.early_stop_settings,
                        best_val_loss=self._early_stop_state.best_val_loss,
                        counter=self._early_stop_state.counter + 1
                    )

                # early stop判定
                if self._early_stop_state.counter >= self._early_stop_state.early_stop_settings.patience:
                    tqdm.write(f"Early stopping at epoch {epoch+1}")
                    break

            # ログの表示
            tqdm.write(f"----- Epoch {epoch+1} -----")
            tqdm.write(f"Train Loss: {train_loss:.4f}, Train Top1 Acc: {train_top1_acc:.4f}, Train Top5 Acc: {train_top5_acc:.4f}, duration: {str(train_end - train_start).split('.')[0]}")
            tqdm.write(f"Val Loss: {val_loss:.4f}, Val Top1 Acc: {val_top1_acc:.4f}, Val Top5 Acc: {val_top5_acc:.4f}, duration: {str(val_end - val_start).split('.')[0]}")

            # lrの更新
            if self._use_lr_scheduler:
                metrics = EpochMetrics(
                    epoch=epoch, 
                    train_loss=train_loss,
                    train_acc=train_top1_acc,
                    val_loss=val_loss,
                    val_acc=val_top1_acc,
                )

                before_scheduler_lr = self._optimizer.param_groups[0]['lr']
                self._lr_scheduler(metrics)
                after_scheduler_lr = self._optimizer.param_groups[0]['lr']
    
                if before_scheduler_lr != after_scheduler_lr:
                    tqdm.write(f"Scheduler step executed at epoch {epoch+1}, lr changed from {before_scheduler_lr} to {after_scheduler_lr}")

    def make_log(self, args: argparse.Namespace):
        '''
        ログの作成
        '''

        # フォルダの作成
        os.makedirs(self._log_dir, exist_ok=True)

        # モデルの保存
        model_path = os.path.join(self._log_dir, "final.pth")
        torch.save(self._model.state_dict(), model_path)
        best_model_path = os.path.join(self._log_dir, "best.pth")
        torch.save(self._best_model_state, best_model_path)

        # 学習結果の推移の保存
        history_filename = os.path.join(self._log_dir, "history.csv")
        with open(history_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train duration", "val duration", "train top1 acc", "train top5 acc", "train loss", "val top1 acc", "val top5 acc", "val loss"])

            for i in range(len(self._train_end_history)):
                train_duration = self._train_end_history[i] - self._train_start_history[i]
                val_duration = self._val_end_history[i] - self._val_start_history[i]

                writer.writerow([
                    i + 1,
                    f"{train_duration}",
                    f"{val_duration}",
                    self._train_top1_acc_history[i],
                    self._train_top5_acc_history[i],
                    self._train_loss_history[i],
                    self._val_top1_acc_history[i],
                    self._val_top5_acc_history[i],
                    self._val_loss_history[i]
                ])

        # argsの保存
        args_filename = os.path.join(self._log_dir, "args.txt")
        with open(args_filename, mode="w") as f:
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")