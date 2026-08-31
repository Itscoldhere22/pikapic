"""
Robustness evaluation script for the AI-generated image detector.

This script loads a trained model checkpoint and evaluates the model on clean
images and images affected by real-world transformations such as JPEG
compression, blur, resizing, noise, color jitter, and cropping. It calculates
the ROC-AUC performance for each transformation, prints a formatted evaluation
report, and generates a bar chart summarizing the model's robustness across
different transformation categories.
"""

import argparse
import sys

import torch
import matplotlib.pyplot as plt
from evaluate import build_model, evaluate_model, SELECTION_VARIANTS, format_report

def plot_robustness_summary(results: dict, output_path: str | None = None):
    """Plot ROC-AUC for clean images and each robustness category."""

    categories = {
        "Clean": ["clean"],
        "JPEG": ["jpeg_90", "jpeg_70", "jpeg_50", "jpeg_30"],
        "Blur": ["blur_0.5", "blur_1.0", "blur_2.0"],
        "Resize": ["resize_0.5", "resize_0.25"],
        "Noise": ["noise_0.02", "noise_0.05", "noise_0.10"],
        "Color Jitter": ["color_jitter"],
        "Crop": ["crop_0.8"],
    }

    names = []
    aucs = []

    for category, variants in categories.items():
        available = [
            results[v]["roc_auc"]
            for v in variants
            if v in results and not math.isnan(results[v]["roc_auc"])
        ]

        if not available:
            continue

        names.append(category)
        aucs.append(np.mean(available) * 100)

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(names, aucs)

    ax.set_ylabel("ROC-AUC (%)")
    ax.set_title("Robustness Evaluation")
    ax.set_ylim(0, 100)

    # Display values above bars
    for bar, value in zip(bars, aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--manifest", default="data/image_manifest.csv", help="Manifest CSV with path/label/split.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-variants", default=",".join(SELECTION_VARIANTS),
                        help="Comma-separated variants for the selection score.")
    # Streamlit passes its own args through sys.argv before `--`; only parse
    # what comes after `--` if present, otherwise fall back to defaults.
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []
    return parser.parse_args(argv)

def load_model(checkpoint_path: str, device_str: str):
    device = torch.device(device_str)
    model = build_model(num_classes=1, pretrained=False).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, device

def main():
    args = parse_cli_args()
    
    model, device = load_model(args.checkpoint, args.device)
    variant_names = [v.strip() for v in args.eval_variants.split(",") if v.strip()]
    results, threshold = evaluate_model(
        model,
        manifest=args.manifest,
        split="val",
        variant_names=variant_names,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    print(format_report(results, threshold))

    plot_robustness_summary(
        results,
        output_path="outputs/robustness_summary.png",
    )
    
if __name__ == "__main__":
    main()
