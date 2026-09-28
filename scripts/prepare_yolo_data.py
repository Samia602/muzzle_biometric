import argparse
import shutil
import zipfile
from pathlib import Path


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


def find_yaml(folder):

    yamls = list(folder.rglob("*.yaml"))

    if len(yamls) == 0:
        return None

    return yamls[0]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zip",
        required=True
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    zip_path = Path(args.zip)

    output_dir = root / "data" / "yolo"

    extract_zip(
        zip_path,
        output_dir
    )

    yaml_file = find_yaml(output_dir)

    if yaml_file is None:

        print("\nERROR: No YAML file found.")

        print("\nFiles found:")

        for p in list(output_dir.rglob("*"))[:50]:
            print(p)

        return

    print("\nDataset extracted successfully.")

    print(f"\nYAML file found:")
    print(yaml_file)

    print("\nFolder structure:")

    for p in list(output_dir.rglob("*"))[:30]:
        print(p)


if __name__ == "__main__":
    main()