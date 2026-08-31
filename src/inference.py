"""Batch inference script for AIGC detection on image directories.

Takes a directory of images and outputs predictions to a JSON file containing
image paths and confidence scores for AI-generated detection.

Usage:
    uv run python src/inference.py \
        --checkpoint checkpoints/best.pth \
        --image-dir /path/to/images \
        --output results.json \
        --device cuda

Output JSON format:
    [
        {"image_path": "path/to/image1.jpg", "pred": 0.92},
        {"image_path": "path/to/image2.jpg", "pred": 0.15},
        ...
    ]

Where pred is a confidence score in [0, 1]:
    - pred ≈ 1.0 → high confidence the image is AI-generated
    - pred ≈ 0.0 → high confidence the image is real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from evaluate import load_checkpoint

# Constants (must match app.py and training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224

# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}


def get_image_paths(image_dir: Path) -> list[Path]:
    """Collect all supported image files from the directory."""
    image_paths = []
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    for ext in SUPPORTED_EXTENSIONS:
        image_paths.extend(sorted(image_dir.glob(f"*{ext}")))
        image_paths.extend(sorted(image_dir.glob(f"*{ext.upper()}")))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for p in image_paths:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            unique_paths.append(p)
    
    return unique_paths


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Preprocess a PIL image to model input (batch of 1)."""
    transform = T.Compose(
        [
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transform(img.convert("RGB")).unsqueeze(0)


@torch.inference_mode()
def predict_batch(
    model: torch.nn.Module,
    images: list[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    """Predict on a batch of PIL images. Returns probabilities in [0, 1]."""
    tensors = [preprocess_image(img).to(device) for img in images]
    batch = torch.cat(tensors, dim=0)
    logits = model(batch)
    probs = torch.sigmoid(logits).squeeze(1)
    return probs.cpu().numpy()


def run_inference(
    checkpoint_path: str,
    image_dir: str,
    output_path: str,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 16,
) -> None:
    """Run inference on all images in a directory and save results to JSON.
    
    Args:
        checkpoint_path: Path to model checkpoint
        image_dir: Directory containing images to classify
        output_path: Path to save results JSON
        device_str: Device to run inference on ("cuda" or "cpu")
        batch_size: Batch size for inference
    """
    device = torch.device(device_str)
    
    # Load model
    print(f"Loading checkpoint from {checkpoint_path}...")
    model = load_checkpoint(checkpoint_path, device)
    model.eval()
    
    # Collect images
    image_dir = Path(image_dir)
    image_paths = get_image_paths(image_dir)
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_paths)} images to process")
    
    # Run inference
    results = []
    failed_images = []
    
    with tqdm(total=len(image_paths), desc="Processing images") as pbar:
        batch_paths = []
        batch_images = []
        
        for image_path in image_paths:
            try:
                # Load and preprocess image
                img = Image.open(image_path)
                batch_images.append(img)
                batch_paths.append(image_path)
                
                # Process batch when full or at end
                if len(batch_images) >= batch_size or image_path == image_paths[-1]:
                    probs = predict_batch(model, batch_images, device)
                    
                    for path, prob in zip(batch_paths, probs):
                        results.append({
                            "image_path": str(path.relative_to(image_dir)),
                            "pred": float(prob),
                        })
                    
                    batch_images = []
                    batch_paths = []
                    pbar.update(batch_size if len(batch_images) == 0 else len(batch_paths))
                
            except Exception as e:
                failed_images.append((str(image_path), str(e)))
                pbar.update(1)
    
    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Processed: {len(results)} images")
    if failed_images:
        print(f"Failed: {len(failed_images)} images")
        for path, error in failed_images[:5]:
            print(f"  - {path}: {error}")
        if len(failed_images) > 5:
            print(f"  ... and {len(failed_images) - 5} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch inference on images in a directory"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing images to classify",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save results JSON file",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)",
    )
    
    args = parser.parse_args()
    
    run_inference(
        checkpoint_path=args.checkpoint,
        image_dir=args.image_dir,
        output_path=args.output,
        device_str=args.device,
        batch_size=args.batch_size,
    )
