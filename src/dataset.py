from pathlib import Path
import random
import pandas as pd


# Project paths
PROJECT_ROOT = Path(r"C:\Users\ntu_a\Projects\Techjam26")
DATASET_ROOT = PROJECT_ROOT / "genimage"

OUTPUT_PATH = PROJECT_ROOT / "pikadet" / "data" / "image_manifest.csv"

SEED = 42
IMAGES_PER_SOURCE = 2_000

random.seed(SEED)


# Adjust these names if your actual folder names differ
GENERATOR_DIRS = {
    "ADM": "adm",
    "BigGAN": "biggan",
    "GLIDE": "glide",
    "Stable Diffusion v1.5": "sdv5",
    "VQDM": "vqdm",
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


def find_images(folder: Path) -> list[Path]:
    """Recursively find image files inside a folder."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sample_images(folder: Path, count: int) -> list[Path]:
    """Randomly sample a fixed number of images from a folder."""
    images = find_images(folder)

    if len(images) < count:
        raise ValueError(
            f"{folder} contains only {len(images)} images, "
            f"but {count} are required."
        )

    return random.sample(images, count)


def add_records(
    records: list[dict],
    images: list[Path],
    label: int,
    source: str,
    split: str = "train",
) -> None:
    """Add image metadata to the manifest."""
    for image_path in images:
        records.append(
            {
                "path": str(image_path),
                "label": label,  # 0 = real, 1 = AI-generated
                "source": source,
                "split": split,
            }
        )


def main() -> None:
    records = []

    for generator_name, folder_name in GENERATOR_DIRS.items():
        generator_root = DATASET_ROOT / folder_name / "train"

        ai_folder = generator_root / "ai"
        real_folder = generator_root / "nature"

        # 2,000 AI-generated images from each generator
        ai_images = sample_images(ai_folder, IMAGES_PER_SOURCE)
        add_records(
            records=records,
            images=ai_images,
            label=1,
            source=generator_name,
        )

        # 2,000 real images from each generator's nature folder
        real_images = sample_images(real_folder, IMAGES_PER_SOURCE)
        add_records(
            records=records,
            images=real_images,
            label=0,
            source=f"{generator_name}_real",
        )

    dataframe = pd.DataFrame(records)

    # Shuffle the complete dataset
    dataframe = dataframe.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved manifest to: {OUTPUT_PATH}")
    print(f"Total images: {len(dataframe)}")
    print("\nClass counts:")
    print(dataframe["label"].value_counts())

    print("\nSource counts:")
    print(dataframe["source"].value_counts())


if __name__ == "__main__":
    main()