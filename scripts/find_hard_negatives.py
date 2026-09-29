import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.models import EmbeddingNet


# -----------------------------
# Settings
# -----------------------------
IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_embedding(model, image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        emb = model(tensor)

    emb = F.normalize(emb, p=2, dim=1)
    return emb[0]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--top", type=int, default=20)

    args = parser.parse_args()

    data_dir = Path(args.data)

    # -----------------------------
    # Load identities
    # -----------------------------
    identities = sorted([
        p for p in data_dir.iterdir()
        if p.is_dir()
    ])

    print(f"Identities available: {len(identities)}")

    # -----------------------------
    # Load model
    # -----------------------------
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu"
    )

    model = EmbeddingNet(
        embedding_dim=checkpoint["embedding_dim"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # -----------------------------
    # Generate embeddings
    # -----------------------------
    print("\nGenerating embeddings...")

    embeddings = {}

    for identity in identities:

        images = [
            p for p in identity.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        embeddings[identity.name] = []

        for image_path in images:

            emb = get_embedding(
                model,
                image_path
            )

            embeddings[identity.name].append(
                (image_path, emb)
            )

    # -----------------------------
    # Generate DIFFERENT-COW pairs
    # -----------------------------
    different_pairs = []

    for _ in range(args.pairs):

        cow_a, cow_b = random.sample(
            identities,
            2
        )

        image_a, emb_a = random.choice(
            embeddings[cow_a.name]
        )

        image_b, emb_b = random.choice(
            embeddings[cow_b.name]
        )

        similarity = F.cosine_similarity(
            emb_a.unsqueeze(0),
            emb_b.unsqueeze(0)
        ).item()

        different_pairs.append(
            (
                similarity,
                cow_a.name,
                image_a,
                cow_b.name,
                image_b
            )
        )

    # -----------------------------
    # Highest similarity first
    # -----------------------------
    different_pairs.sort(
        key=lambda x: x[0],
        reverse=True
    )

    print("\n================================")
    print("MOST SIMILAR DIFFERENT-COW PAIRS")
    print("================================")

    for i, (
        similarity,
        cow_a,
        image_a,
        cow_b,
        image_b
    ) in enumerate(
        different_pairs[:args.top],
        1
    ):

        print(f"\n{i}. Similarity: {similarity:.4f}")
        print(f"   Cow A: {cow_a}")
        print(f"   Image: {image_a}")
        print(f"   Cow B: {cow_b}")
        print(f"   Image: {image_b}")


if __name__ == "__main__":
    main()