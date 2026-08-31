"""Greedy model soup for the AIGC detector.

Adapted from the standard greedy-soup recipe (Wortsman et al., "Model
Soups") to this codebase's checkpoint format: each checkpoint saved by
train.py is a dict with a "model" state_dict plus metadata, including
"selection_score" (0.4*clean_auc + 0.6*trans_auc) for the epoch it was
saved at. We reuse that stored score to rank ingredients instead of
reading a separate results file, then greedily average state_dicts and
re-evaluate on the held-out val split after each addition.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from evaluate import SELECTION_VARIANTS, build_model, evaluate_model, selection_score


def parse_args():
    parser = argparse.ArgumentParser(description="Greedy soup over AIGC detector checkpoints.")
    parser.add_argument(
        "checkpoints", nargs="+",
        help="Paths to the .pt checkpoints to soup, e.g. 4 of them: "
             "checkpoints/v1/best.pt checkpoints/v2/best.pt ...",
    )
    parser.add_argument("--manifest", required=True, help="Manifest CSV (same as training).")
    parser.add_argument("--out", default="checkpoints/greedy_soup.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-variants", default=",".join(SELECTION_VARIANTS))
    return parser.parse_args()


def get_model_from_sd(state_dict, device):
    model = build_model(num_classes=1, pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_sd(state_dict, args, device, variant_names):
    model = get_model_from_sd(state_dict, device)
    results, _ = evaluate_model(
        model,
        manifest=args.manifest,
        split="val",
        variant_names=variant_names,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    return selection_score(results)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    variant_names = [v.strip() for v in args.eval_variants.split(",") if v.strip()]

    if len(args.checkpoints) < 2:
        raise SystemExit("Need at least 2 checkpoints to soup.")

    # Rank ingredients by the score already stored at checkpoint time,
    # same role as the ImageNet2p ranking in the reference script.
    ranked = []
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if "model" not in ckpt:
            raise SystemExit(f"{path} has no 'model' key — is this a raw state_dict, not a train.py checkpoint?")
        score = ckpt.get("selection_score", -math.inf)
        ranked.append((path, ckpt, score))
    ranked.sort(key=lambda x: x[2], reverse=True)

    print("Ranked ingredients (by stored selection_score):")
    for path, _, score in ranked:
        print(f"  {path}: {score:.4f}")

    # Start the soup with the strongest single checkpoint, then re-evaluate
    # it fresh on the held-out val split (the stored score may be from an
    # older manifest/eval-variant config, so don't trust it as the baseline).
    soup_paths = [ranked[0][0]]
    soup_sd = {k: v.clone() for k, v in ranked[0][1]["model"].items()}
    best_val_score = evaluate_sd(soup_sd, args, device, variant_names)
    print(f"\nStarting soup = [{ranked[0][0]}], held-out score={best_val_score:.4f}")

    # Greedily test each remaining checkpoint.
    for path, ckpt, _ in ranked[1:]:
        n = len(soup_paths)
        candidate_sd = {
            k: soup_sd[k].clone() * (n / (n + 1.0)) + ckpt["model"][k].clone() * (1.0 / (n + 1.0))
            for k in soup_sd
        }
        candidate_score = evaluate_sd(candidate_sd, args, device, variant_names)
        print(f"Testing {path}: candidate score={candidate_score:.4f}, best so far={best_val_score:.4f}")

        if candidate_score > best_val_score:
            soup_paths.append(path)
            soup_sd = candidate_sd
            best_val_score = candidate_score
            print(f"  -> added. Soup is now {soup_paths}")
        else:
            print("  -> skipped.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": soup_sd, "selection_score": best_val_score, "ingredients": soup_paths},
        out_path,
    )
    print(f"\nFinal soup: {soup_paths}")
    print(f"Held-out score: {best_val_score:.4f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()