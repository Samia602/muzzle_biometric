from pathlib import Path
import json
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import EmbeddingNet, ArcFaceHead

# ============================================================
# PATHS
# ============================================================

# Auto-detect root directory (works locally on Windows/Linux or Google Colab)
if Path("/content").exists():
    ROOT = Path("/content/muzzle_biometric")
else:
    ROOT = Path(r"C:\muzzle_biometric")

TRAIN_DIR = ROOT / "data" / "arcface" / "train"
VAL_DIR = ROOT / "data" / "arcface" / "val"
TEST_DIR = ROOT / "data" / "arcface" / "test"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# IMAGE TYPES
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================
# DATASET
# ============================================================

class CowDataset(Dataset):
    def __init__(self, root, transform=None, class_to_idx=None):
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(
                f"\nDataset folder not found:\n{self.root}"
            )

        folders = [f for f in self.root.iterdir() if f.is_dir()]
        folders.sort(key=lambda x: x.name)

        # Map classes (reuse training mapping for validation to ensure consistency)
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = {
                folder.name: index for index, folder in enumerate(folders)
            }

        self.samples = []
        for folder in folders:
            if folder.name in self.class_to_idx:
                label = self.class_to_idx[folder.name]
                for image in folder.iterdir():
                    if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                        self.samples.append((image, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# ============================================================
# TRANSFORMS
# ============================================================

def create_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(8),
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.10
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def create_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ============================================================
# COUNT SPLIT
# ============================================================

def get_split_info(directory):
    if not directory.exists():
        return 0, 0
    identities = [f for f in directory.iterdir() if f.is_dir()]
    image_count = sum(
        1 for id_dir in identities for img in id_dir.iterdir()
        if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS
    )
    return len(identities), image_count

# ============================================================
# TRAIN & EVALUATE FUNCTIONS
# ============================================================

def train_epoch(model, head, loader, optimizer, criterion, device):
    model.train()
    head.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        embeddings = model(images)
        logits = head(embeddings, labels)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate_epoch(model, head, loader, criterion, device):
    model.eval()
    head.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        embeddings = model(images)
        logits = head(embeddings, labels)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += images.size(0)

    return (total_loss / total, correct / total) if total > 0 else (0.0, 0.0)

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 50)
    print("ArcFace Cow Muzzle Training")
    print("=" * 50)

    # Device configuration (Uses CUDA/GPU in Colab, falls back to CPU locally)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing Device: {device}")
    if device.type == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    # Check directories
    print("\nChecking dataset directories...")
    for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not directory.exists():
            print(f"Warning: Directory missing - {directory}")
        else:
            print(f"Found: {directory}")

    # Dataset stats
    train_classes, train_images = get_split_info(TRAIN_DIR)
    val_classes, val_images = get_split_info(VAL_DIR)
    test_classes, test_images = get_split_info(TEST_DIR)

    print("\nDataset Summary:")
    print(f"Train:      {train_classes} identities, {train_images} images")
    print(f"Validation: {val_classes} identities, {val_images} images")
    print(f"Test:       {test_classes} identities, {test_images} images")

    # Datasets & Loaders
    train_transform = create_train_transform()
    val_transform = create_val_transform()

    train_dataset = CowDataset(TRAIN_DIR, transform=train_transform)
    val_dataset = CowDataset(
        VAL_DIR, transform=val_transform, class_to_idx=train_dataset.class_to_idx
    )

    batch_size = 32 if device.type == "cuda" else 16
    num_workers = 2 if device.type == "cuda" else 0

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    ) if len(val_dataset) > 0 else None

    # Model & Head setup
    embedding_dim = 512
    model = EmbeddingNet(embedding_dim=embedding_dim).to(device)
    head = ArcFaceHead(
        embedding_dim=embedding_dim,
        num_classes=len(train_dataset.class_to_idx),
        scale=30.0,
        margin=0.5
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=0.0003,
        weight_decay=0.0001
    )

    epochs = 40
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training Loop
    history = []
    best_val_loss = float("inf")
    checkpoint_path = MODEL_DIR / "arcface_muzzle.pt"

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, head, train_loader, optimizer, criterion, device
        )

        if val_loader:
            val_loss, val_acc = validate_epoch(
                model, head, val_loader, criterion, device
            )
        else:
            val_loss, val_acc = 0.0, 0.0

        scheduler.step()

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc * 100:.2f}%"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        # Save checkpoint (Save best if validation exists, otherwise save latest)
        is_best = val_loader and (val_loss < best_val_loss)
        if is_best or not val_loader:
            if is_best:
                best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "embedding_dim": embedding_dim,
                "num_classes": len(train_dataset.class_to_idx),
                "class_to_idx": train_dataset.class_to_idx,
                "model_state_dict": model.state_dict(),
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "history": history
            }
            torch.save(checkpoint, checkpoint_path)

    # Save Metadata JSON
    metadata = {
        "architecture": "ResNet18",
        "embedding_dimension": embedding_dim,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.0003,
        "train_identities": train_classes,
        "train_images": train_images,
        "val_identities": val_classes,
        "val_images": val_images,
        "final_train_loss": train_loss,
        "final_train_accuracy": train_acc,
        "best_val_loss": best_val_loss if val_loader else 0.0,
        "checkpoint": str(checkpoint_path)
    }

    with open(MODEL_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print(f"Model saved to: {checkpoint_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()