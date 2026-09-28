import argparse
import random
import shutil
import zipfile
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def extract_zip(zip_path, output_dir):

    print(f"\nExtracting: {zip_path}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)

    print("Extraction complete.")


def find_identity_folders(root):

    identities = []

    for directory in root.rglob("*"):

        if not directory.is_dir():
            continue

        images = [
            p for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if len(images) >= 2:
            identities.append(
                (directory, images)
            )

    return identities


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zip",
        required=True
    )

    parser.add_argument(
        "--crop",
        action="store_true"
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    zip_path = Path(args.zip)

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    extracted = (
        root
        / "data"
        / "arcface_extracted"
    )

    output = (
        root
        / "data"
        / "arcface"
    )

    extract_zip(
        zip_path,
        extracted
    )

    identities = find_identity_folders(
        extracted
    )

    print(
        f"\nCandidate identity folders found: "
        f"{len(identities)}"
    )

    if not identities:

        print(
            "\nERROR: No identity folders found."
        )

        print(
            "\nFirst extracted directories:"
        )

        for p in list(extracted.rglob("*"))[:100]:
            print(p)

        return

    # Remove duplicates caused by nested folders.
    cleaned = {}

    for folder, images in identities:

        key = str(folder.resolve())

        cleaned[key] = (
            folder,
            images
        )

    identities = list(cleaned.values())

    # Keep identities with at least two images.
    identities = [
        x for x in identities
        if len(x[1]) >= 2
    ]

    print(
        f"Usable identities: {len(identities)}"
    )

    print("\nExample identities:")

    for folder, images in identities[:10]:

        print(
            f"{folder.name}: "
            f"{len(images)} images"
        )

    random.seed(42)

    random.shuffle(identities)

    total = len(identities)

    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    train_ids = identities[:train_end]
    val_ids = identities[train_end:val_end]
    test_ids = identities[val_end:]

    if output.exists():
        shutil.rmtree(output)

    for split in ["train", "val", "test"]:
        (output / split).mkdir(
            parents=True,
            exist_ok=True
        )

    def copy_identity(split, folder, images):

        destination = (
            output
            / split
            / folder.name
        )

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        for image in images:

            shutil.copy2(
                image,
                destination / image.name
            )

    for split, items in [
        ("train", train_ids),
        ("val", val_ids),
        ("test", test_ids),
    ]:

        print(
            f"\n{split}: "
            f"{len(items)} identities"
        )

        for folder, images in items:

            copy_identity(
                split,
                folder,
                images
            )

    print("\n================================")
    print("ArcFace dataset preparation done")
    print("================================")

    print(
        f"Training identities: "
        f"{len(train_ids)}"
    )

    print(
        f"Validation identities: "
        f"{len(val_ids)}"
    )

    print(
        f"Testing identities: "
        f"{len(test_ids)}"
    )

    print(
        f"\nOutput: {output}"
    )

    if args.crop:

        print(
            "\nNOTE:"
        )

        print(
            "--crop was requested."
        )

        print(
            "YOLO cropping should be enabled only "
            "after models/yolo_muzzle.pt exists."
        )


if __name__ == "__main__":
    main()