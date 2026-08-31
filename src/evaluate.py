"""Evaluation for the AIGC detector.

Provides numpy metric helpers (ROC-AUC, balanced accuracy, precision/recall/F1,
FPR, threshold tuning) and :func:`evaluate_model`, which runs a model over the
clean split plus every deterministic robustness variant. The module also owns
model construction/checkpoint loading so :mod:`src.train` can reuse it without a
circular import.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50

from dataset import (
    AIGCDataset,
    VARIANT_DISTORTIONS,
    build_eval_transform,
    build_variant_transform,
)

# Representative one-per-category subset used for lightweight checkpoint
# selection during training (full breakdown uses all variants).
SELECTION_VARIANTS = [
    "jpeg_70",
    "blur_1.0",
    "resize_0.5",
    "noise_0.05",
    "color_jitter",
    "crop_0.8",
]

# Category rollup: mean ROC-AUC over each group (mirrors PLAN.md breakdown).
CATEGORIES = {
    "clean": ["clean"],
    "jpeg": ["jpeg_90", "jpeg_70", "jpeg_50", "jpeg_30"],
    "blur": ["blur_0.5", "blur_1.0", "blur_2.0"],
    "resize": ["resize_0.5", "resize_0.25"],
    "noise": ["noise_0.02", "noise_0.05", "noise_0.10"],
    "color_jitter": ["color_jitter"],
    "crop": ["crop_0.8"],
    "extras": [
        "flip",
        "rotate_10",
        "autocontrast",
        "equalize",
        "sharpen",
        "posterize_5",
        "median_3",
        "webp_75",
        "saltpepper",
    ],
}


# ---------------------------------------------------------------------------
# Metrics (pure numpy)
# ---------------------------------------------------------------------------


def roc_auc(labels, scores) -> float:
    """Area under the ROC curve via the trapezoidal rule."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(labels)) < 2:
        return float("nan")

    order = np.argsort(scores)[::-1]
    labels = labels[order]
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    tpr = np.concatenate([[0.0], np.cumsum(labels) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1 - labels) / n_neg])
    return float(np.trapezoid(tpr, fpr))


def tune_threshold(labels, scores) -> float:
    """Threshold maximizing Youden's J (TPR - FPR) on the given split."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    best_j, best_t = -1.0, 0.5
    for t in np.unique(scores):
        pred = scores >= t
        tp = int(np.logical_and(pred, pos).sum())
        fp = int(np.logical_and(pred, ~pos).sum())
        j = tp / n_pos - fp / n_neg
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def compute_metrics(labels, scores, threshold: float) -> dict:
    """Compute a full set of classification metrics at a fixed threshold."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    preds = (scores >= threshold).astype(int)

    tp = int(np.logical_and(preds == 1, labels == 1).sum())
    fp = int(np.logical_and(preds == 1, labels == 0).sum())
    tn = int(np.logical_and(preds == 0, labels == 0).sum())
    fn = int(np.logical_and(preds == 0, labels == 1).sum())

    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    tpr = tp / max(1, tp + fn)  # recall / sensitivity
    tnr = tn / max(1, tn + fp)  # specificity
    precision = tp / max(1, tp + fp)
    recall = tpr
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    fpr = fp / max(1, fp + tn)

    return {
        "roc_auc": roc_auc(labels, scores),
        "balanced_accuracy": float((tpr + tnr) / 2.0),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "threshold": float(threshold),
    }


def transformed_auc(results: dict) -> float:
    """Mean ROC-AUC over all non-clean variants."""
    aucs = [
        results[k]["roc_auc"]
        for k in results
        if k != "clean" and not math.isnan(results[k]["roc_auc"])
    ]
    return float(np.mean(aucs)) if aucs else float("nan")


def selection_score(results: dict) -> float:
    """0.4 * clean AUC + 0.6 * transformed AUC (PLAN.md selection score)."""
    clean = results["clean"]["roc_auc"]
    trans = transformed_auc(results)
    if math.isnan(clean) or math.isnan(trans):
        return float("-inf")
    return 0.4 * clean + 0.6 * trans


# ---------------------------------------------------------------------------
# Model construction / loading
# ---------------------------------------------------------------------------


def build_model(num_classes: int = 1, pretrained: bool = True):
    """ResNet-50 with a binary classifier head (ImageNet-pretrained by default)."""
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(path, device):
    # Pretrained weights are overwritten by the checkpoint, so skip the download.
    model = build_model(num_classes=1, pretrained=False)

    # # Allow numpy
    # torch.serialization.add_safe_globals([torch.serialization.Storage]) # Often required alongside numpy
    # torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])

    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _predict(model, dataset, device, batch_size, num_workers):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    labels_list, scores_list = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        scores = torch.sigmoid(logits).squeeze(1)
        scores_list.append(scores.cpu().numpy())
        labels_list.append(labels.numpy())
    return np.concatenate(labels_list), np.concatenate(scores_list)


def evaluate_model(
    model,
    manifest,
    split,
    variant_names,
    device,
    threshold=None,
    batch_size=64,
    num_workers=0,
) -> tuple[dict, float]:
    """Evaluate a model on clean + each named variant; return results and threshold.

    The threshold is tuned on the clean split when not provided.
    """
    model.eval()

    clean_ds = AIGCDataset(manifest, split=split, transform=build_eval_transform())
    labels, scores = _predict(model, clean_ds, device, batch_size, num_workers)
    if threshold is None:
        threshold = tune_threshold(labels, scores)

    results = {"clean": compute_metrics(labels, scores, threshold)}
    for name in variant_names:
        ds = AIGCDataset(manifest, split=split, transform=build_variant_transform(name))
        vlabels, vscores = _predict(model, ds, device, batch_size, num_workers)
        results[name] = compute_metrics(vlabels, vscores, threshold)

    return results, threshold


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------


def category_rollup(results: dict) -> pd.DataFrame:
    """Mean ROC-AUC (and balanced accuracy) per transformation category."""
    rows = []
    for category, names in CATEGORIES.items():
        present = [n for n in names if n in results]
        if not present:
            continue
        auc = np.mean([results[n]["roc_auc"] for n in present])
        bacc = np.mean([results[n]["balanced_accuracy"] for n in present])
        rows.append({"category": category, "roc_auc": auc, "balanced_accuracy": bacc})
    return pd.DataFrame(rows)


def format_report(results: dict, threshold: float) -> str:
    df = pd.DataFrame(results).T
    cols = [
        "roc_auc",
        "balanced_accuracy",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr",
    ]
    df = df[cols].round(4)
    rollup = category_rollup(results)

    header = f"Threshold (tuned on clean val): {threshold:.4f}\n"
    return header + "\nPer-variant metrics:\n" + df.to_string() + "\n\nCategory rollup (mean AUC):\n" + rollup.to_string(index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an AIGC detector checkpoint.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with path/label/split.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint file.")
    parser.add_argument("--split", default="val", help="Manifest split to evaluate.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold (else tuned on clean).")
    parser.add_argument("--variants", default=None, help="Comma-separated variant names (default: all).")
    parser.add_argument("--output", default=None, help="Optional CSV path for per-variant metrics.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device)

    if args.variants:
        variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        variant_names = list(VARIANT_DISTORTIONS)

    results, threshold = evaluate_model(
        model,
        manifest=args.manifest,
        split=args.split,
        variant_names=variant_names,
        device=device,
        threshold=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    print(format_report(results, threshold))

    if args.output:
        pd.DataFrame(results).T.round(4).to_csv(args.output)
        print(f"\nWrote per-variant metrics to {args.output}")


if __name__ == "__main__":
    main()
