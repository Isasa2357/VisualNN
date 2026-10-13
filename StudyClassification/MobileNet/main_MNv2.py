# MobileNet v2の学習スクリプト

import argparse
from tqdm import tqdm
import os

import torch
from torch import optim, nn
from torchvision import transforms

from comm.dataset import Dataset, make_dataloader, get_dataset_enum, get_class_num
from comm.train_eval import train_loop, make_log

from MobileNet.network import MobileNetV2

def main():
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    # model 
    parser.add_argument('--user-model', type=str, help='Path to a custom user-defined model', default=None)
    parser.add_argument('--drop-rate', type=float, default=0.0, help='Dropout rate of the MobileNetV2')
    parser.add_argument('--expand_rate', type=float, default=6.0, help='Expansion rate of the MobileNetV2')
    parser.add_argument('--param-dtype', type=str, default='float32', help='Parameter data type for the model')

    # train
    parser.add_argument('--dataset', type=str, default='imagenette2-160')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)

    # dataloader
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for the dataloader')

    # device and project settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--project', type=str, default='run')
    parser.add_argument('--result', type=str, default='MobileNetV2')
    parser.add_argument('--no-robocopy', action='store_true', help='Use robocopy to copy the dataset')

    args = parser.parse_args()

    user_model = args.user_model
    drop_rate = args.drop_rate
    expand_rate = args.expand_rate
    if args.param_dtype == 'float16':
        param_dtype = torch.float16
    elif args.param_dtype == 'float32':
        param_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported parameter data type: {args.param_dtype}")
    dataset_enum = get_dataset_enum(args.dataset)
    class_num = get_class_num(dataset_enum)
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    dataloader_num_workers = args.num_workers
    device = torch.device(args.device)
    project = args.project
    result = args.result
    no_robocopy = args.no_robocopy

    model = MobileNetV2(
        num_classes=class_num,
        expand_rate=expand_rate, 
        dropout_rate=drop_rate
    ).to(device, dtype=param_dtype)
    
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

    train_loader, val_loader = make_dataloader(dataset_enum, batch_size, train_transform, val_transform, True, dataloader_num_workers, robocopy=not no_robocopy)

    # 学習の実行
    train_losses, train_accs, val_losses, val_accs, timestamps = train_loop(model, train_loader, val_loader, criterion, optimizer, device, epochs)

    # 結果の保存
    make_log(
        os.path.join(project, result),
        model, 
        train_losses, 
        train_accs, 
        val_losses, 
        val_accs, 
        timestamps, 
        args
    )

    print("Done")

if __name__ == "__main__":
    main()