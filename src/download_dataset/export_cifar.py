from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100


def export_to_images(dataset_class, source_dir: Path, output_dir: Path, name: str):
    print(f"\nExporting {name}...")

    for split, train in [("train", True), ("test", False)]:
        dataset = dataset_class(
            root=source_dir,
            train=train,
            download=False,
        )

        split_dir = output_dir / split

        print(f"  {split}: {len(dataset)} images")

        for i, (image, label) in enumerate(dataset):
            class_name = dataset.classes[label]

            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            filename = class_dir / f"{i:06d}.png"
            image.save(filename)

            if (i + 1) % 5000 == 0:
                print(f"    {i + 1}/{len(dataset)}")

    print(f"{name} completed: {output_dir}")


def main():
    root_dir = Path(r"C:\work\datasets")

    # CIFAR-10
    export_to_images(
        dataset_class=CIFAR10,
        source_dir=root_dir / "CIFAR10",
        output_dir=root_dir / "CIFAR10_images",
        name="CIFAR-10",
    )

    # CIFAR-100
    export_to_images(
        dataset_class=CIFAR100,
        source_dir=root_dir / "CIFAR100",
        output_dir=root_dir / "CIFAR100_images",
        name="CIFAR-100",
    )


if __name__ == "__main__":
    main()