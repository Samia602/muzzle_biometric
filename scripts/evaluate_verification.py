import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.models import EmbeddingNet


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def load_images(folder):

    return [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in IMAGE_EXTENSIONS
    ]


def make_transform():

    return transforms.Compose([
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


def get_embedding(
    model,
    image_path,
    transform
):

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    ).unsqueeze(0)

    with torch.no_grad():

        embedding = model(
            tensor
        )

    return embedding[0]


def cosine_similarity(a, b):

    return F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0)
    ).item()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="models/arcface_muzzle.pt"
    )

    parser.add_argument(
        "--data",
        default="data/arcface/test"
    )

    parser.add_argument(
        "--pairs",
        type=int,
        default=1000
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    checkpoint_path = (
        root / args.checkpoint
    )

    data_dir = (
        root / args.data
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model = EmbeddingNet(
        embedding_dim=512
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    transform = make_transform()

    identities = []

    for folder in data_dir.iterdir():

        if not folder.is_dir():
            continue

        images = load_images(folder)

        if len(images) >= 2:
            identities.append(
                (folder.name, images)
            )

    print(
        "Identities available:",
        len(identities)
    )

    if len(identities) < 2:

        raise RuntimeError(
            "Not enough identities."
        )

    embeddings = {}

    print(
        "\nGenerating embeddings..."
    )

    for identity, images in identities:

        embeddings[identity] = []

        for image in images:

            emb = get_embedding(
                model,
                image,
                transform
            )

            embeddings[identity].append(
                emb
            )

    random.seed(42)

    same_scores = []
    different_scores = []

    # --------------------------------------------
    # SAME COW PAIRS
    # --------------------------------------------

    for identity, embs in embeddings.items():

        if len(embs) < 2:
            continue

        for _ in range(
            max(1, args.pairs // 2)
        ):

            a, b = random.sample(
                embs,
                2
            )

            score = cosine_similarity(
                a,
                b
            )

            same_scores.append(
                score
            )

            if len(same_scores) >= args.pairs:
                break

        if len(same_scores) >= args.pairs:
            break

    # --------------------------------------------
    # DIFFERENT COW PAIRS
    # --------------------------------------------

    identity_names = list(
        embeddings.keys()
    )

    for _ in range(args.pairs):

        id1, id2 = random.sample(
            identity_names,
            2
        )

        a = random.choice(
            embeddings[id1]
        )

        b = random.choice(
            embeddings[id2]
        )

        score = cosine_similarity(
            a,
            b
        )

        different_scores.append(
            score
        )

    print(
        "\nSame-cow pairs:",
        len(same_scores)
    )

    print(
        "Different-cow pairs:",
        len(different_scores)
    )

    # --------------------------------------------
    # FIND THRESHOLD
    # --------------------------------------------

    thresholds = np.linspace(
        -1,
        1,
        1001
    )

    best_threshold = 0.5
    best_accuracy = 0.0

    for threshold in thresholds:

        same_correct = sum(
            s >= threshold
            for s in same_scores
        )

        different_correct = sum(
            s < threshold
            for s in different_scores
        )

        total = (
            len(same_scores)
            +
            len(different_scores)
        )

        accuracy = (
            same_correct
            +
            different_correct
        ) / total

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_threshold = threshold

    print(
        "\n================================"
    )

    print(
        f"Best threshold: "
        f"{best_threshold:.4f}"
    )

    print(
        f"Verification accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()