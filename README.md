# Pikapic — Robust AI-Generated Image Detection

A prototype that tells **AI-generated images** apart from **real (camera) images**,
and keeps working after images are re-processed in the wild — compressed, blurred,
resized, cropped, re-colored, or noised.

It trains an ImageNet-pretrained **ResNet-50** binary classifier (≈23.5M params,
well under the 2B limit) with strong-but-probabilistic robustness augmentation,
then evaluates it **per transformation** and on **unseen generators**.

---

## What you'll find here

| File | What it does |
|---|---|
| `src/dataset.py` | Builds the dataset manifest (list of images) **and** loads/augments images during training & evaluation. |
| `src/train.py` | Trains the ResNet-50 model and saves checkpoints. |
| `src/evaluate.py` | Loads a checkpoint and reports metrics, broken down by transformation. |

There are three steps: **build the manifest → train → evaluate**. Each is one command.

---

## 1. Requirements

- **Python 3.11** (see `.python-version`)
- **`uv`** — the package manager used here. Install: <https://docs.astral.sh/uv/>
- A **GPU is strongly recommended** (the defaults assume an RTX 4060-class GPU), but
  everything also runs on CPU with `--device cpu` (much slower).
- Internet access the **first time** you train, so torchvision can download the
  ImageNet ResNet-50 weights (~100 MB, cached afterwards).

Dependencies (`pandas`, `torch`, `torchaudio`, `torchvision`) are already listed in
`pyproject.toml`; `uv` installs them for you.

---

## 2. Setup (one-time)

From the **project root**:

```bash
uv sync
```

This creates a `.venv` and installs all dependencies. After this, run every command
below prefixed with `uv run` (so it uses that environment):

```bash
uv run python src/train.py --help
```

> If you already have a Python 3.11 environment with `torch`/`torchvision`/`pandas`
> installed, you can use plain `python src/...` instead of `uv run python src/...`.

---

## 3. Your data layout

The code expects the **GenImage** dataset, laid out with one folder per generator
family, and inside each a `train/ai` (generated) and `train/nature` (real) folder:

```
/path/to/genimage/
├── adm/train/ai/…          # AI images from the "ADM" generator
├── adm/train/nature/…      # real images paired with ADM
├── biggan/train/ai/…
├── biggan/train/nature/…
├── glide/…
├── midjourney/…
├── sdv14/…
├── sdv15/…
├── vqdm/…
└── wukong/…
```

Two things you should edit in `src/dataset.py` (top of the file) to match your data:

1. **`GENERATOR_DIRS`** — maps a display name → the actual folder name.
   ```python
   GENERATOR_DIRS = {
       "ADM": "adm",
       "BigGAN": "biggan",
       ...
   }
   ```
2. **`HELD_OUT_FAMILIES`** — the two families reserved for *unseen-generator testing*
   (never used in training):
   ```python
   HELD_OUT_FAMILIES = {"Wukong", "Midjourney"}
   ```

If your real images live in a differently-named folder (e.g. `real` instead of
`nature`), change `"nature"` in `build_manifest()` accordingly.

---

## 4. Step 1 — Build the manifest

This scans the folders, samples images, assigns the train/val/test split, and writes
a CSV that the other two scripts read.

```bash
uv run python src/dataset.py --dataset-root /path/to/genimage
```

It prints a summary and writes `data/image_manifest.csv` by default. The CSV has one
row per image:

| column | meaning |
|---|---|
| `path` | absolute path to the image file |
| `label` | `0` = real, `1` = AI-generated |
| `source` | generator family name |
| `family` | generator family (same as `source`) |
| `split` | `train`, `val`, or `test` |

**How the split works:** the two `HELD_OUT_FAMILIES` become `test` (unseen
generators). The remaining families become `train`, with a small balanced `val`
carved out.

**How many images:** by default ~10,000 AI + ~10,000 real across the training
families, plus a 2,000-image validation set. Change these with flags (see the
reference below) or by editing `N_AI_TRAIN` / `N_REAL_TRAIN` / `N_VAL` in
`src/dataset.py`.

---

## 5. Step 2 — Train

```bash
uv run python src/train.py --manifest data/image_manifest.csv
```

What happens:

1. **Stage 1** — only the classifier head is trained for a few epochs (fast warm-up).
2. **Stage 2** — the full network is fine-tuned.
3. After each epoch, the model is evaluated on the **clean + transformed** validation
   set and scored. A checkpoint is saved every epoch, and the best one is copied to
   `checkpoints/best.pt`.
4. Training stops early if the score doesn't improve for a few epochs.

Output files (in `checkpoints/` by default):

- `head_epoch_1.pt`, `head.pt` — head-only stage snapshots (before fine-tuning)
- `epoch_1.pt`, `epoch_2.pt`, … — every fine-tuning epoch's checkpoint
- `best.pt` — the best checkpoint (use this for evaluation)

Stage-2 checkpoints (`epoch_*.pt`, `best.pt`) carry the **full resume state** — model
weights, optimizer, AMP scaler, the epoch counter, and RNG state — so you can stop and
continue training exactly where you left off.

**First run tip** — verify everything works on a tiny subset before a real run:

```bash
uv run python src/train.py --manifest data/image_manifest.csv --sanity --no-pretrained --device cpu --workers 0
```

`--sanity` limits to a handful of images, `--no-pretrained` skips the weight download,
`--device cpu` avoids needing a GPU. It should finish in seconds. If it does, drop
those flags for the real run.

**Resuming a run** — training can be stopped (Ctrl-C / timeout) and continued later:

```bash
uv run python src/train.py --manifest data/image_manifest.csv \
  --resume checkpoints/epoch_5.pt --epochs 8
```

`--resume` loads the model weights **and** the optimizer, AMP scaler, epoch counter,
and RNG state, then continues fine-tuning from the next epoch (so the example above
runs epochs 6–8) and skips the head-only stage. Set `--epochs` to the *total* number
of fine-tuning epochs you want (8 above), or raise it to train longer than originally
planned. If the checkpoint is an older, weights-only file, `--resume` degrades to a
warm start — it loads the weights and begins a fresh optimizer at epoch 1.

---

## 6. Step 3 — Evaluate

```bash
uv run python src/evaluate.py --manifest data/image_manifest.csv --checkpoint checkpoints/best.pt --split val
```

This prints:

1. **The threshold** (tuned on the clean validation set — not a fixed 0.5).
2. **Per-variant metrics** — one row per robustness variant (JPEG q90, blur σ2.0, …).
3. **Category rollup** — mean ROC-AUC per transformation family.

Reported metrics: **ROC-AUC, balanced accuracy, accuracy, precision, recall, F1,
and false-positive rate (FPR)**.

To evaluate on the unseen-generator test set instead:

```bash
uv run python src/evaluate.py \
  --manifest data/image_manifest.csv \
  --checkpoint checkpoints/best.pt \
  --split test
```

To save the per-variant table to a CSV for your report:

```bash
uv run python src/evaluate.py \
  --manifest data/image_manifest.csv \
  --checkpoint checkpoints/best.pt \
  --split val \
  --output results/val_metrics.csv
```

---

## 7. CLI reference

### `src/dataset.py` — build the manifest

| Flag | Default | Description |
|---|---|---|
| `--dataset-root` | *(required)* | Root of the GenImage dataset. Or set the `GENIMAGE_ROOT` env var. |
| `--output` | `data/image_manifest.csv` | Where to write the manifest CSV. |
| `--num-ai` | `10000` | Total AI images across training families. |
| `--num-real` | `10000` | Total real images across training families. |
| `--num-val` | `2000` | Validation images (balanced by family & class). |
| `--seed` | `42` | Random seed for sampling. |

### `src/train.py` — train the model

| Flag | Default | Description |
|---|---|---|
| `--manifest` | *(required)* | Path to the manifest CSV. |
| `--out-dir` | `checkpoints` | Where checkpoints are saved. |
| `--epochs` | `8` | Fine-tuning epochs (stage 2). |
| `--head-epochs` | `2` | Head-only epochs (stage 1). |
| `--batch-size` | `64` | Batch size. Lower if you run out of GPU memory. |
| `--max-per-class` | `10000` | Cap on images per class (10k AI + 10k real). |
| `--lr` | `1e-4` | Learning rate for the backbone (stage 2). |
| `--head-lr` | `1e-3` | Learning rate for the classifier head. |
| `--weight-decay` | `1e-4` | AdamW weight decay. |
| `--aug-strength` | `1.0` | Scales the train augmentation probability (`1.0` = default; `<1` lighter, `>1` heavier). |
| `--pos-weight` | *(off)* | Positive-class weight for `BCEWithLogitsLoss` (use if your classes are imbalanced). |
| `--workers` | `4` | DataLoader worker processes. |
| `--device` | `cuda` (or `cpu`) | `cuda`, `cpu`, or a specific device like `cuda:0`. |
| `--seed` | `42` | Random seed. |
| `--patience` | `30` | Early-stopping patience (epochs). |
| `--eval-variants` | see below | Comma-separated variants used for the selection score. |
| `--no-amp` | off | Disable mixed precision. |
| `--no-pretrained` | off | Train from scratch (skips the weight download; for offline/testing). |
| `--resume` | *(none)* | Path to a checkpoint to resume from — restores weights, optimizer, scaler, epoch & RNG, then continues fine-tuning (skips the head-only stage). |
| `--sanity` | off | Tiny run to smoke-test the pipeline. |

### `src/evaluate.py` — evaluate a checkpoint

| Flag | Default | Description |
|---|---|---|
| `--manifest` | *(required)* | Path to the manifest CSV. |
| `--checkpoint` | *(required)* | Path to a checkpoint file (e.g. `checkpoints/best.pt`). |
| `--split` | `val` | Which manifest split to evaluate (`val` or `test`). |
| `--device` | `cuda` (or `cpu`) | `cuda` or `cpu`. |
| `--batch-size` | `64` | Batch size. |
| `--workers` | `4` | DataLoader worker processes. |
| `--threshold` | *(auto)* | Override the decision threshold. By default it's tuned on the clean split. |
| `--variants` | all | Comma-separated variant names to evaluate (default: every variant). |
| `--output` | *(none)* | Optional CSV path to write the per-variant table. |
| `--seed` | `0` | Random seed. |

---

## 8. Robustness variants

`--variants` (evaluate) and `--eval-variants` (train) accept these names,
comma-separated with no spaces, e.g. `--variants jpeg_70,blur_1.0,crop_0.8`.

**Required (challenge subset)**

| Variant | What it does |
|---|---|
| `jpeg_90`, `jpeg_70`, `jpeg_50`, `jpeg_30` | JPEG re-compression at that quality |
| `blur_0.5`, `blur_1.0`, `blur_2.0` | Gaussian blur at that sigma |
| `resize_0.5`, `resize_0.25` | Downscale to 50% / 25%, then upscale back |
| `noise_0.02`, `noise_0.05`, `noise_0.10` | Gaussian noise at that sigma |
| `color_jitter` | Brightness / contrast / saturation ±20% |
| `crop_0.8` | Center crop to 80% |

**Proposed extras**

| Variant | What it does |
|---|---|
| `flip` | Horizontal flip |
| `rotate_10` | Rotate 10° |
| `autocontrast` | Auto-contrast |
| `equalize` | Histogram equalization |
| `sharpen` | Unsharp-mask sharpening (×2.0) |
| `posterize_5` | 5-bit posterize |
| `median_3` | 3×3 median filter |
| `webp_75` | WebP re-encode at quality 75 |
| `saltpepper` | Salt & pepper (impulse) noise |

**Category rollup** in the report groups these as:

- `jpeg` = mean over `jpeg_90/70/50/30`
- `blur` = mean over `blur_0.5/1.0/2.0`
- `resize` = mean over `resize_0.5/0.25`
- `noise` = mean over `noise_0.02/0.05/0.10`
- plus `color_jitter`, `crop`, `extras`, and `clean`.

---

## 9. How the "best" checkpoint is chosen

After each epoch the model is scored as:

```
score = 0.4 × (clean ROC-AUC) + 0.6 × (transformed ROC-AUC)
```

where *transformed ROC-AUC* is the mean AUC over the robustness variants. By default
the selection uses a lightweight one-per-category subset (`jpeg_70, blur_1.0,
resize_0.5, noise_0.05, color_jitter, crop_0.8`) to keep training fast; change it with
`--eval-variants`. The checkpoint with the highest score wins and is saved as
`best.pt`. Early stopping also keys off this score.

---

## 10. Training augmentations (why it's robust)

During training, ~25% of images are kept **clean**; the rest get **one or two**
random distortions drawn from a pool (JPEG, blur, downscale, color jitter, sharpness,
posterize, median filter, WebP, salt & pepper, Gaussian noise, crop). This teaches the
model to recognize images through realistic re-processing **without** destroying every
forensic signal in every image.

---

## 11. Troubleshooting

- **`ModuleNotFoundError: No module named 'dataset'`** — run the scripts as
  `python src/train.py …` (from the project root), **not** `python -m src.train`.
  The scripts use flat imports on purpose.
- **Out of GPU memory** — lower `--batch-size` (e.g. `--batch-size 32`).
- **"Folder does not exist"** during `dataset.py` — check `GENERATOR_DIRS` names and
  that each family has `train/ai` and `train/nature` subfolders.
- **No GPU / offline** — use `--device cpu` and `--no-pretrained` for a quick check.
- **Slow first run** — the first train downloads ImageNet weights; after that they're
  cached.
- **`--workers` errors on Windows** — set `--workers 0`.

---

## 12. Roadmap (from `PLAN.md`)

This is **Phase 1** (reliable baseline). Later phases: hyperparameter variants
(Phase 2), multi-crop inference (Phase 3), and greedy weight-space model soup
(Phase 4) — see `PLAN.md` for details.

---

## 13. Phase 2 — hyperparameter variants

Phase 2 (from `PLAN.md`) is a small sweep over the Phase 1 baseline: run 3–5
variants, record every configuration, and keep their checkpoints separate so the
Phase 4 greedy soup can average the best ones. **You don't need new scripts** —
every axis is already a CLI flag. The two that were missing are `--aug-strength`
and `--pos-weight` (added to `src/train.py` and `src/dataset.py`).

Here is every Phase 2 knob and where it lives:

| Axis | Flag | Default | Where it's defined |
|---|---|---|---|
| Backbone learning rate | `--lr` | `1e-4` | `parse_args()` in `src/train.py` |
| Head learning rate | `--head-lr` | `1e-3` | `parse_args()` in `src/train.py` |
| Weight decay | `--weight-decay` | `1e-4` | `parse_args()` in `src/train.py` |
| Random seed | `--seed` | `42` | `parse_args()` in `src/train.py` |
| **Augmentation strength** | `--aug-strength` | `1.0` | `build_train_transform()` in `src/dataset.py` (scales `p = 0.5 * aug_strength`) |
| **Class weighting** | `--pos-weight` | *(off)* | `BCEWithLogitsLoss` in `src/train.py` |

**One rule:** give every variant its **own `--out-dir`**, otherwise each run
overwrites the previous run's `checkpoints/best.pt`.

The "Medium" / "Strong" augmentation labels map to a concrete `--aug-strength`
value. Each distortion gate fires with probability `p = 0.5 × aug_strength`, and
the share of images left **clean** is `(1 − p)²`:

| Label | `--aug-strength` | gate p | clean | one | two |
|---|---|---|---|---|---|
| Medium | `1.0` | 0.50 | 25% | 50% | 25% |
| Strong | `1.5` | 0.75 | ~6% | ~38% | ~56% |

(`Strong` is aggressive — ~6% clean is below PLAN's "keep 20–30% clean" guidance.
Use `--aug-strength 1.2` ≈ 16% clean if you want to stay inside that window.)

A concrete 3-variant sweep (run one at a time):

```bash
# 1. Conservative fine-tuning
uv run python src/train.py --manifest data/image_manifest.csv --out-dir checkpoints/v1_conservative \
  --head-lr 5e-4 --lr 5e-5 --weight-decay 1e-4 --aug-strength 1.0 --seed 42

# 2. Stronger adaptation
uv run python src/train.py --manifest data/image_manifest.csv --out-dir checkpoints/v2_stronger \
  --head-lr 1e-3 --lr 2e-4 --weight-decay 1e-4 --aug-strength 1.0 --seed 42

# 3. Regularized / imbalance-robust
uv run python src/train.py --manifest data/image_manifest.csv --out-dir checkpoints/v3_regularized \
  --head-lr 1e-3 --lr 1e-4 --weight-decay 5e-4 --aug-strength 1.5 --seed 42
```

> On class weighting: the dataset is balanced (10k AI + 10k real), so
> `--pos-weight` has nothing to fix. Only add it (e.g. `--pos-weight 1.5`) if you
> deliberately imbalance the real/AI counts.

After each run, evaluate its `best.pt`:

```bash
uv run python src/evaluate.py --manifest data/image_manifest.csv --checkpoint checkpoints/v1_conservative/best.pt --split val
```

Each run's checkpoint stores its full config (now including `aug_strength` and
`pos_weight`), and the final `Best score=` line prints the selection score. Keep
both so you can rank the variants and feed the winners into the Phase 4 soup.
