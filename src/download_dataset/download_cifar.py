from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100


def download_dataset(dataset_class, save_dir: Path, name: str):
    print(f"Downloading {name}...")

    train_dataset = dataset_class(
        root=save_dir,
        train=True,
        download=True,
    )

    test_dataset = dataset_class(
        root=save_dir,
        train=False,
        download=True,
    )

    print(f"{name} downloaded.")
    print(f"  Save dir : {save_dir}")
    print(f"  Train    : {len(train_dataset)} images")
    print(f"  Test     : {len(test_dataset)} images")
    print()


def main():
    root_dir = Path(r"C:\work\datasets")

    download_dataset(
        CIFAR10,
        root_dir / "CIFAR10",
        "CIFAR-10",
    )

    download_dataset(
        CIFAR100,
        root_dir / "CIFAR100",
        "CIFAR-100",
    )


if __name__ == "__main__":
    main()