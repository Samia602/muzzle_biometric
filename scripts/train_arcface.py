import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.models import EmbeddingNet, ArcFaceHead


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


class CowDataset(Dataset):

    def __init__(
        self,
        root,
        transform=None,
        class_to_idx=None
    ):

        self.root = Path(root)

        self.transform = transform

        folders = [
            p for p in self.root.iterdir()
            if p.is_dir()
        ]

        folders.sort(
            key=lambda x: x.name
        )

        if class_to_idx is None:

            self.class_to_idx = {
                folder.name: i
                for i, folder in enumerate(folders)
            }

        else:

            self.class_to_idx = class_to_idx

        self.samples = []

        for folder in folders:

            if folder.name not in self.class_to_idx:
                continue

            label = self.class_to_idx[
                folder.name
            ]

            for image in folder.iterdir():

                if (
                    image.is_file()
                    and image.suffix.lower()
                    in IMAGE_EXTENSIONS
                ):

                    self.samples.append(
                        (
                            image,
                            label
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def create_transforms():

    train_transform = transforms.Compose([
        transforms.Resize(
            (224, 224)
        ),

        transforms.RandomRotation(
            8
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    return train_transform, eval_transform


def train_epoch(
    backbone,
    head,
    loader,
    optimizer,
    device
):

    backbone.train()
    head.train()

    total_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        embeddings = backbone(images)

        logits = head(
            embeddings,
            labels
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += images.size(0)

    return (
        total_loss / total,
        correct / total
    )


def evaluate_classification(
    backbone,
    head,
    loader,
    device
):

    backbone.eval()
    head.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            embeddings = backbone(images)

            logits = head(
                embeddings,
                labels
            )

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += images.size(0)

    return (
        total_loss / total,
        correct / total
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=40
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003
    )

    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    root = Path(__file__).resolve().parents[1]

    train_dir = (
        root
        / "data"
        / "arcface"
        / "train"
    )

    val_dir = (
        root
        / "data"
        / "arcface"
        / "val"
    )

    model_dir = (
        root
        / "models"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device("cpu")

    print("\n================================")
    print("ArcFace Training")
    print("================================")

    print("Device:", device)
    print("Architecture: ResNet18")
    print("Embedding: 512")
    print("Batch:", args.batch)
    print("Epochs:", args.epochs)

    train_transform, eval_transform = (
        create_transforms()
    )

    train_dataset = CowDataset(
        train_dir,
        transform=train_transform
    )

    class_to_idx = (
        train_dataset.class_to_idx
    )

    # Validation only uses identities that
    # exist in training for classification evaluation.
    #
    # The separate verification stage will evaluate
    # unseen identities.
    val_dataset = CowDataset(
        val_dir,
        transform=eval_transform,
        class_to_idx=class_to_idx
    )

    print(
        "Training images:",
        len(train_dataset)
    )

    print(
        "Training classes:",
        len(class_to_idx)
    )

    print(
        "Validation images:",
        len(val_dataset)
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "No training images found."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0
    )

    backbone = EmbeddingNet(
        embedding_dim=512
    ).to(device)

    head = ArcFaceHead(
        embedding_dim=512,
        num_classes=len(class_to_idx),
        scale=30.0,
        margin=0.5
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters())
        +
        list(head.parameters()),
        lr=args.lr,
        weight_decay=0.0001
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    best_val_loss = float("inf")

    checkpoint_path = (
        model_dir
        / "arcface_muzzle.pt"
    )

    for epoch in range(
        1,
        args.epochs + 1
    ):

        train_loss, train_acc = (
            train_epoch(
                backbone,
                head,
                train_loader,
                optimizer,
                device
            )
        )

        val_loss, val_acc = (
            evaluate_classification(
                backbone,
                head,
                val_loader,
                device
            )
            if len(val_dataset) > 0
            else (0.0, 0.0)
        )

        scheduler.step()

        print(
            f"\nEpoch {epoch}/{args.epochs}"
        )

        print(
            f"Train loss: {train_loss:.4f}"
        )

        print(
            f"Train accuracy: "
            f"{train_acc * 100:.2f}%"
        )

        print(
            f"Val loss: {val_loss:.4f}"
        )

        print(
            f"Val accuracy: "
            f"{val_acc * 100:.2f}%"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint = {
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "class_to_idx": class_to_idx,
                "embedding_dim": 512,
                "architecture": "resnet18",
                "epoch": epoch,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }

            torch.save(
                checkpoint,
                checkpoint_path
            )

            print(
                "Saved best checkpoint."
            )

    print("\n================================")
    print("Training finished.")
    print("================================")

    print(
        "Checkpoint:",
        checkpoint_path
    )

    metadata_path = (
        model_dir
        / "arcface_metadata.json"
    )

    metadata = {
        "architecture": "resnet18",
        "embedding_dim": 512,
        "num_classes": len(class_to_idx),
        "checkpoint": str(
            checkpoint_path
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=4
        ),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()