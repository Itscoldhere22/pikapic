"""Dataset loading, manifest building, and augmentation for AIGC detection.

This module has two responsibilities:

1. **Manifest building** (``main``): sample images from the GenImage directory
   tree, assign a family-disjoint train/val/test split, and write a CSV with
   columns ``path, label, source, family, split``. This is normally run once.

2. **Dataset loading + augmentation** (:class:`AIGCDataset`): consume a manifest
   (CSV path or DataFrame) and yield ``(image_tensor, label)`` pairs, with
   online augmentation for the train split and deterministic single-transform
   variants for robustness evaluation.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

# ---------------------------------------------------------------------------
# Manifest-building configuration
# ---------------------------------------------------------------------------

# Adjust these names if your actual folder names differ. Each generator family
# has a ``train/ai`` (generated) and ``train/nature`` (real) subfolder.
GENERATOR_DIRS = {
    "ADM": "adm",
    "BigGAN": "biggan",
    "GLIDE": "glide",
    "Midjourney": "midjourney",
    "Stable Diffusion v1.5": "sdv5",
    "VQDM": "vqdm",
    "Wukong": "wukong",
}

# These families are held out completely for unseen-generator testing.
HELD_OUT_FAMILIES = {"Wukong", "Midjourney"}

# Initial training size (PLAN.md "Dataset plan"): ~10k AI + ~10k real/nature.
# These are the total counts across *training* families; val is carved out on
# top of them. Edit here or via --num-ai / --num-real CLI flags.
N_AI_TRAIN = 10_000
N_REAL_TRAIN = 10_000
N_VAL = 2_000

SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ImageNet normalization constants (ResNet-50 pretrained weights expect these).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------


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
            f"{folder} contains only {len(images)} images, but {count} are required."
        )
    return random.sample(images, count)


def add_records(
    records: list[dict],
    images: list[Path],
    label: int,
    source: str,
    family: str | None = None,
    split: str = "train",
) -> None:
    """Add image metadata to the manifest."""
    family = family or source
    for image_path in images:
        records.append(
            {
                "path": str(image_path),
                "label": label,  # 0 = real, 1 = AI-generated
                "source": source,
                "family": family,
                "split": split,
            }
        )


def sample_balanced_val(df: pd.DataFrame, n_val: int, seed: int) -> pd.DataFrame:
    """Sample a family- and label-balanced validation subset from a DataFrame."""
    groups = list(df.groupby(["family", "label"]))
    per_group = math.ceil(n_val / len(groups))
    parts = []
    for _, group in groups:
        n = min(len(group), per_group)
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts)


def build_manifest(
    dataset_root: Path,
    num_ai: int,
    num_real: int,
    num_val: int,
    seed: int,
) -> pd.DataFrame:
    """Build a manifest DataFrame with family-disjoint train/val/test splits."""
    random.seed(seed)

    train_families = [name for name in GENERATOR_DIRS if name not in HELD_OUT_FAMILIES]
    per_family_ai = math.ceil(num_ai / len(train_families))
    per_family_real = math.ceil(num_real / len(train_families))

    records: list[dict] = []

    # Training + validation families.
    for name in train_families:
        root = dataset_root / GENERATOR_DIRS[name] / "train"
        ai_images = sample_images(root / "ai", per_family_ai)
        real_images = sample_images(root / "nature", per_family_real)
        add_records(records, ai_images, label=1, source=name, family=name)
        add_records(records, real_images, label=0, source=name, family=name)

    # Held-out families -> unseen-generator test.
    for name in HELD_OUT_FAMILIES:
        root = dataset_root / GENERATOR_DIRS[name] / "train"
        ai_images = sample_images(root / "ai", per_family_ai)
        real_images = sample_images(root / "nature", per_family_real)
        add_records(records, ai_images, label=1, source=name, family=name, split="test")
        add_records(records, real_images, label=0, source=name, family=name, split="test")

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Carve a balanced validation subset out of the training rows.
    train_mask = df["split"] == "train"
    val_idx = sample_balanced_val(df[train_mask], num_val, seed).index
    df.loc[val_idx, "split"] = "val"

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the image manifest CSV.")
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("GENIMAGE_ROOT"),
        help="Root of the GenImage dataset (contains per-generator folders).",
    )
    parser.add_argument("--output", default="data/image_manifest.csv")
    parser.add_argument("--num-ai", type=int, default=N_AI_TRAIN)
    parser.add_argument("--num-real", type=int, default=N_REAL_TRAIN)
    parser.add_argument("--num-val", type=int, default=N_VAL)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not args.dataset_root:
        parser.error("--dataset-root is required (or set GENIMAGE_ROOT).")

    df = build_manifest(
        dataset_root=Path(args.dataset_root),
        num_ai=args.num_ai,
        num_real=args.num_real,
        num_val=args.num_val,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved manifest to: {output_path}")
    print(f"Total images: {len(df)}")
    print("\nSplit counts:")
    print(df["split"].value_counts())
    print("\nClass counts:")
    print(df["label"].value_counts())
    print("\nFamily counts:")
    print(df["family"].value_counts())


# ---------------------------------------------------------------------------
# Custom PIL-based transforms
# ---------------------------------------------------------------------------
# Each of these accepts a PIL image and returns a PIL image. They are reused by
# both the train augmentation pool and the deterministic eval variants. Using
# PIL directly (rather than torchvision) gives exact control over the challenge
# parameters (JPEG quality, Gaussian sigma, resize scale, ...).


def _as_range(value, name):
    if isinstance(value, (tuple, list)):
        return (float(value[0]), float(value[1]))
    return (float(value), float(value))


class JPEG:
    """Re-encode as JPEG at a fixed quality or a uniform range."""

    def __init__(self, quality):
        self.quality = _as_range(quality, "quality")

    def __call__(self, img):
        q = random.randint(int(self.quality[0]), int(self.quality[1]))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class GaussianBlur:
    """PIL Gaussian blur with an exact sigma (or uniform sigma range)."""

    def __init__(self, sigma):
        self.sigma = _as_range(sigma, "sigma")

    def __call__(self, img):
        s = random.uniform(self.sigma[0], self.sigma[1])
        return img.filter(ImageFilter.GaussianBlur(radius=s))


class Downscale:
    """Downscale by a factor then upscale back to the original size."""

    def __init__(self, scale):
        self.scale = _as_range(scale, "scale")

    def __call__(self, img):
        s = random.uniform(self.scale[0], self.scale[1])
        w, h = img.size
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        small = img.resize((nw, nh), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class GaussianNoise:
    """Additive Gaussian noise, sigma on the [0, 1] intensity scale."""

    def __init__(self, sigma):
        self.sigma = _as_range(sigma, "sigma")

    def __call__(self, img):
        s = random.uniform(self.sigma[0], self.sigma[1])
        arr = np.asarray(img).astype(np.float32) / 255.0
        noise = np.random.normal(0.0, s, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr)


class SaltAndPepper:
    """Impulse noise: replace a fraction of pixels with black or white."""

    def __init__(self, amount):
        self.amount = _as_range(amount, "amount")

    def __call__(self, img):
        p = random.uniform(self.amount[0], self.amount[1])
        arr = np.asarray(img).copy()
        h, w = arr.shape[:2]
        n = int(p * h * w)
        if n == 0:
            return img
        for c in range(arr.shape[2]):
            idx = np.random.choice(h * w, n, replace=False)
            arr[..., c].reshape(-1)[idx] = np.random.choice([0, 255], n)
        return Image.fromarray(arr)


class MedianFilter:
    """Median filter (denoising / beautification)."""

    def __init__(self, size=3):
        self.size = size

    def __call__(self, img):
        return img.filter(ImageFilter.MedianFilter(size=self.size))


class Sharpen:
    """Unsharp-mask sharpening with a fixed factor or uniform range."""

    def __init__(self, factor):
        self.factor = _as_range(factor, "factor")

    def __call__(self, img):
        f = random.uniform(self.factor[0], self.factor[1])
        return ImageEnhance.Sharpness(img).enhance(f)


class Posterize:
    """Reduce color bit-depth to a fixed value or uniform range."""

    def __init__(self, bits):
        self.bits = _as_range(bits, "bits")

    def __call__(self, img):
        b = random.randint(int(self.bits[0]), int(self.bits[1]))
        return ImageOps.posterize(img, b)


class WebPReencode:
    """Re-encode as WebP (falls back to JPEG if WebP is unsupported)."""

    def __init__(self, quality):
        self.quality = _as_range(quality, "quality")

    def __call__(self, img):
        q = random.randint(int(self.quality[0]), int(self.quality[1]))
        try:
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=q)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        except (OSError, ValueError):
            q = max(10, min(q, 95))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            return Image.open(buf).convert("RGB")


class Crop80:
    """Crop the central (or a random) 80% of each dimension."""

    def __init__(self, center=True):
        self.center = center

    def __call__(self, img):
        w, h = img.size
        cw, ch = int(w * 0.8), int(h * 0.8)
        if self.center:
            left, top = (w - cw) // 2, (h - ch) // 2
        else:
            left = random.randint(0, max(0, w - cw))
            top = random.randint(0, max(0, h - ch))
        return img.crop((left, top, left + cw, top + ch))


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------


def _distortion_pool() -> T.RandomChoice:
    """One random robustness distortion, chosen uniformly from the pool."""
    return T.RandomChoice(
        [
            JPEG((30, 90)),
            GaussianBlur((0.5, 2.0)),
            Downscale((0.5, 0.9)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            Sharpen((0.5, 2.0)),
            T.RandomAutocontrast(p=1.0),
            T.RandomEqualize(p=1.0),
            Posterize((4, 6)),
            MedianFilter(size=3),
            WebPReencode((60, 90)),
            SaltAndPepper((0.01, 0.05)),
            GaussianNoise((0.02, 0.10)),
            Crop80(center=False),
        ]
    )


def _to_model(size: int) -> list:
    return [
        T.Resize((size, size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def build_train_transform(size: int = 224, aug_strength: float = 1.0) -> T.Compose:
    """Probabilistic robustness augmentation for training.

    Two independent ``RandomApply`` gates yield ~25% clean images, ~50% with one
    distortion, and ~25% with two at ``aug_strength=1.0`` — per PLAN.md, avoiding
    uniform max-strength distortion that would erase forensic signals.

    ``aug_strength`` scales the per-gate probability ``p = 0.5 * aug_strength``
    (clamped to [0, 1]): ``1.0`` = default, ``<1`` = lighter augmentation (more
    clean images), ``>1`` = heavier (fewer clean, more double distortions).
    """
    p = min(1.0, max(0.0, 0.5 * aug_strength))
    return T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [
                    T.RandomRotation(
                        degrees=10,
                        interpolation=T.InterpolationMode.BILINEAR,
                        fill=0,
                    )
                ],
                p=0.3,
            ),
            T.RandomApply([_distortion_pool()], p=p),
            T.RandomApply([_distortion_pool()], p=p),
        ]
        + _to_model(size)
    )


def build_eval_transform(size: int = 224) -> T.Compose:
    """Clean transform (resize + normalize) for validation/testing."""
    return T.Compose(_to_model(size))


# Deterministic single-transform variants for the robustness breakdown. Values
# follow Question-Statement.md; extras are our additions.
VARIANT_DISTORTIONS = {
    # Required (challenge subset)
    "jpeg_90": JPEG(90),
    "jpeg_70": JPEG(70),
    "jpeg_50": JPEG(50),
    "jpeg_30": JPEG(30),
    "blur_0.5": GaussianBlur(0.5),
    "blur_1.0": GaussianBlur(1.0),
    "blur_2.0": GaussianBlur(2.0),
    "resize_0.5": Downscale(0.5),
    "resize_0.25": Downscale(0.25),
    "noise_0.02": GaussianNoise(0.02),
    "noise_0.05": GaussianNoise(0.05),
    "noise_0.10": GaussianNoise(0.10),
    "color_jitter": T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    "crop_0.8": Crop80(center=True),
    # Proposed extras
    "flip": T.RandomHorizontalFlip(p=1.0),
    "rotate_10": T.RandomRotation(
        degrees=(10, 10), interpolation=T.InterpolationMode.BILINEAR, fill=0
    ),
    "autocontrast": T.RandomAutocontrast(p=1.0),
    "equalize": T.RandomEqualize(p=1.0),
    "sharpen": Sharpen(2.0),
    "posterize_5": Posterize(5),
    "median_3": MedianFilter(3),
    "webp_75": WebPReencode(75),
    "saltpepper": SaltAndPepper(0.02),
}


def build_variant_transform(name: str, size: int = 224) -> T.Compose:
    """Deterministic transform for a single named robustness variant."""
    if name not in VARIANT_DISTORTIONS:
        raise KeyError(f"Unknown variant '{name}'. Known: {list(VARIANT_DISTORTIONS)}")
    return T.Compose([VARIANT_DISTORTIONS[name]] + _to_model(size))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_manifest(manifest) -> pd.DataFrame:
    """Load a manifest from a CSV path or return an existing DataFrame."""
    if isinstance(manifest, (str, Path)):
        return pd.read_csv(manifest)
    return manifest


def cap_per_class(df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    """Cap the number of examples per class (0/1) without rebuilding the manifest."""
    parts = []
    for _, group in df.groupby("label"):
        if len(group) > max_per_class:
            group = group.sample(n=max_per_class, random_state=seed)
        parts.append(group)
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


class AIGCDataset(Dataset):
    """PyTorch Dataset over a manifest of image paths and labels.

    Args:
        manifest: CSV path or pandas DataFrame with columns ``path`` and
            ``label`` (and optionally ``split``).
        split: If the manifest has a ``split`` column, keep only rows matching
            this value (e.g. ``"train"`` / ``"val"`` / ``"test"``).
        transform: Explicit transform. If ``None``, a transform is chosen from
            ``split`` (train -> augmentation, otherwise clean).
        max_per_class: Optional cap on examples per class (e.g. 10000 -> 10k AI
            + 10k real), applied after the split filter.
        aug_strength: Scales the train augmentation probability (1.0 = default,
            <1 lighter / more clean, >1 heavier). Ignored if ``transform`` is set.
    """

    def __init__(
        self,
        manifest,
        split: str | None = "train",
        transform=None,
        max_per_class: int | None = None,
        aug_strength: float = 1.0,
    ):
        df = load_manifest(manifest)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split]
        if max_per_class is not None:
            df = cap_per_class(df, max_per_class, seed=SEED)
        self.df = df.reset_index(drop=True)
        self.transform = transform if transform is not None else self._default_transform(split, aug_strength)

    @staticmethod
    def _default_transform(split, aug_strength=1.0):
        return build_train_transform(aug_strength=aug_strength) if split == "train" else build_eval_transform()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = int(row["label"])
        return img, label


if __name__ == "__main__":
    main()
