"""Training for the AIGC detector (Phase 1: reliable baseline).

Trains an ImageNet-pretrained ResNet-50 binary classifier in two stages
(classifier head first, then full fine-tune), with mixed-precision CUDA
training, per-epoch checkpointing, and early stopping on a robustness-weighted
validation score (PLAN.md): ``0.4 * clean AUC + 0.6 * transformed AUC``.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import AIGCDataset
from evaluate import SELECTION_VARIANTS, build_model, evaluate_model, selection_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_duration(seconds: float) -> str:
    """Human-readable duration, e.g. ``1h 05m 12s`` or ``0.4s``."""
    if seconds < 1:
        return f"{seconds:.1f}s"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the AIGC detector (Phase 1).")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with path/label/split.")
    parser.add_argument("--out-dir", default="checkpoints", help="Directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=8, help="Fine-tuning epochs (stage 2).")
    parser.add_argument("--head-epochs", type=int, default=2, help="Head-only epochs (stage 1).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=10_000, help="Cap examples/class (10k AI + 10k real).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Backbone LR (stage 2).")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Classifier-head LR.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience (epochs).")
    parser.add_argument("--eval-variants", default=",".join(SELECTION_VARIANTS),
                        help="Comma-separated variants for the selection score.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--no-pretrained", action="store_true", help="Train from scratch (offline/quick tests).")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from (skips the head-only stage).")
    parser.add_argument("--sanity", action="store_true", help="Tiny run to smoke-test wiring.")
    return parser.parse_args()


def build_optimizer(model, args, stage: str):
    if stage == "head":
        return torch.optim.AdamW(
            model.fc.parameters(), lr=args.head_lr, weight_decay=args.weight_decay
        )
    backbone_params = [
        p for n, p in model.named_parameters() if not n.startswith("fc.")
    ]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": model.fc.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / max(1, n)


def save_checkpoint(model, path, **meta):
    torch.save({"model": model.state_dict(), **meta}, path)


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    set_seed(args.seed)

    if args.sanity:
        args.epochs = 1
        args.head_epochs = 1
        args.max_per_class = 4

    device = torch.device(args.device)
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = AIGCDataset(
        args.manifest, split="train", max_per_class=args.max_per_class
    )
    val_ds = AIGCDataset(args.manifest, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Train examples: {len(train_ds)}, Val examples: {len(val_ds)}")

    resume = args.resume is not None
    resume_best = -math.inf
    model = build_model(
        num_classes=1, pretrained=(not args.no_pretrained) and not resume
    ).to(device)
    if resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        resume_best = state.get("selection_score", -math.inf)
        print(f"Resumed weights from {args.resume}; skipping the head-only stage.")

    criterion = nn.BCEWithLogitsLoss()

    variant_names = [v.strip() for v in args.eval_variants.split(",") if v.strip()]

    # --- Stage 1: train the classifier head only ---
    if not resume:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        optimizer = build_optimizer(model, args, stage="head")

        print(f"Stage 1: head-only training for {args.head_epochs} epoch(s).")
        head_durations = []
        for epoch in range(1, args.head_epochs + 1):
            t0 = time.perf_counter()
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
            head_durations.append(time.perf_counter() - t0)
            eta = (sum(head_durations) / len(head_durations)) * (args.head_epochs - epoch)
            print(
                f"  [head] epoch {epoch}/{args.head_epochs}  loss={loss:.4f}  "
                f"({format_duration(head_durations[-1])})  ETA ~{format_duration(eta)}"
            )
            save_checkpoint(
                model, out_dir / f"head_epoch_{epoch}.pt", stage="head", epoch=epoch, loss=loss
            )

        # Snapshot of the finished head-stage model, before unfreezing the backbone.
        save_checkpoint(model, out_dir / "head.pt", stage="head", epoch=args.head_epochs, loss=loss)

    # --- Stage 2: fine-tune the full network ---
    for p in model.parameters():
        p.requires_grad = True
    optimizer = build_optimizer(model, args, stage="full")

    best_score = resume_best
    best_epoch = -1
    stale = 0
    print(f"Stage 2: full fine-tuning for {args.epochs} epoch(s).")
    epoch_durations = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        t_train = time.perf_counter() - t0

        results, threshold = evaluate_model(
            model,
            manifest=args.manifest,
            split="val",
            variant_names=variant_names,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        t_epoch = time.perf_counter() - t0
        t_eval = t_epoch - t_train
        epoch_durations.append(t_epoch)

        score = selection_score(results)
        clean_auc = results["clean"]["roc_auc"]
        trans_auc = math.nan if score == -math.inf else (
            score - 0.4 * clean_auc
        ) / 0.6

        eta = (sum(epoch_durations) / len(epoch_durations)) * (args.epochs - epoch)
        print(
            f"  [full] epoch {epoch}/{args.epochs}  loss={loss:.4f}  "
            f"clean_auc={clean_auc:.4f}  trans_auc={trans_auc:.4f}  score={score:.4f}  "
            f"train {format_duration(t_train)}  eval {format_duration(t_eval)}  "
            f"ETA ~{format_duration(eta)}"
        )

        meta = {
            "epoch": epoch,
            "selection_score": score,
            "threshold": threshold,
            "config": {
                "max_per_class": args.max_per_class,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "head_lr": args.head_lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
            },
        }
        save_checkpoint(model, out_dir / f"epoch_{epoch}.pt", **meta)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            save_checkpoint(model, out_dir / "best.pt", **meta)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.")
                break

    print(f"Training finished. Best score={best_score:.4f} at epoch {best_epoch}.")
    print(f"Best checkpoint: {out_dir / 'best.pt'}")
    print(f"Total time: {format_duration(time.perf_counter() - start)}")


if __name__ == "__main__":
    main()
