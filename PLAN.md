We have a shared implementation direction. The attached documents contributed the process and challenge requirements; your confirmed choices define the plan below.

## Recommended system

Use a pretrained ResNet-50 binary classifier:

1. Whole-image input at 224×224.
2. Strong but probabilistic robustness augmentation.
3. Generator-disjoint evaluation.
4. Optional multi-crop patch inference.
5. Greedy weight-space model soup.
6. One final checkpoint selected by robustness-weighted validation performance.

The challenge explicitly emphasizes robustness to compression, blur, resizing, noise, color changes, and cropping. The cited CNN research also supports preprocessing and augmentation for generalization to unseen generators, but transformed data should be carefully separated between training, validation, and testing. [Paper](https://arxiv.org/abs/1912.11035)

## Dataset plan

Start with 20,000 training images total:

- 10,000 real
- 10,000 AIGC
- Approximately 1,667 generated images from each of 6 training generator families
- Two generator families held out completely for unseen-generator testing

Select families for diversity:

- Two diffusion families
- Two GAN/adversarial families
- One image-to-image/translation family
- One visually or architecturally distinct family

Choose real images that resemble the generated images in subject, domain, resolution, aspect ratio, and visual style. Avoid pairing one unrelated real-image source against many generated sources.

Use:

- 2,000-image validation set from the six training families
- 2,000–4,000-image clean test set
- Transformed copies of validation/test images for robustness testing
- Deduplication or grouping of near-duplicates before splitting

When expanding to 100,000 images, preserve class and family balance.

## Data split

Use three evaluation views:

| Evaluation set | Purpose |
|---|---|
| Clean validation | Check ordinary classification performance |
| Transformed validation | Select robust checkpoints and soup members |
| Unseen-generator test | Measure generalization beyond training generators |

Do not use transformed copies from training in validation or testing. Generate test transformations independently.

## Augmentation

Keep roughly 20–30% of training images clean. For the rest, apply one or two transformations probabilistically:

- JPEG compression: quality 30–90
- Blur: σ 0.5–2.0
- Resize down and upscale
- Center/random crop
- Color jitter
- Gaussian noise

Use strong augmentation, but avoid applying every maximum-strength transformation to every image. That would create unrealistic examples and may erase useful forensic signals.

## Training sequence

### Phase 1: reliable baseline

Train ImageNet-pretrained ResNet-50 at 224×224.

- Train classifier head first for a few epochs.
- Then unfreeze and fine-tune the full network.
- Use mixed-precision CUDA training.
- Train for approximately 5–10 epochs initially.
- Save a checkpoint every epoch.
- Use early stopping based on robustness-weighted validation performance.

Suggested starting values:

- Optimizer: AdamW
- Head learning rate: `1e-3`
- Backbone learning rate: `1e-4`
- Weight decay: `1e-4`
- Batch size: largest stable size that fits your RTX 4060
- Loss: binary cross-entropy with logits
- Input: 224×224

### Phase 2: focused variants

Run only 3–5 variants:

- Learning rate
- Augmentation strength
- Weight decay
- Random seed
- Optional class weighting

Record every configuration and dataset split.

### Phase 3: patch/multi-crop inference

Do not initially train a separate patch model.

For each image, calculate:

- One whole-image prediction
- Center crop prediction
- Four corner/random crop predictions

Average the outputs. Use this only if it improves the held-out robustness validation set.

This provides patch-level evidence while avoiding noisy patch labels and a second training pipeline.

### Phase 4: greedy soup

Use the best compatible checkpoints from the hyperparameter runs.

1. Rank checkpoints by validation score.
2. Start with the best checkpoint.
3. Tentatively average its weights with the next candidate.
4. Keep the candidate only if the robustness-weighted score improves.
5. Continue until no candidate improves performance.
6. Evaluate the final soup once on the unseen-generator test set.

Use only models with identical architecture, preprocessing, and classifier head.

Your selection score should be:

```text
0.4 × clean ROC-AUC
+ 0.6 × transformed/unseen-generator ROC-AUC
```

Do not inspect the final unseen-generator test results while making soup decisions.

## Evaluation

Report:

- ROC-AUC
- Balanced accuracy
- Accuracy
- Precision
- Recall
- F1
- False-positive rate

Break results down by:

- Clean images
- JPEG compression
- Blur
- Resize
- Noise
- Color jitter
- Cropping
- Unseen generator family

Use a threshold tuned on validation data rather than automatically assuming 0.5.

## Timeline

### 29 August evening

- Inspect GenImage folder structure.
- Select generator families.
- Build a manifest with paths, labels, family IDs, and split IDs.
- Implement dataset loading and augmentations.
- Run a small batch sanity check.

### 30 August

- Train the whole-image baseline.
- Verify validation and transformed-test evaluation.
- Create the demo inference path.
- Save the best baseline checkpoint.

### 31 August

- Run 3–5 focused variants.
- Attempt greedy soup.
- Add multi-crop inference if time permits.
- Produce transformation-wise evaluation tables.

### 1 September morning

- Freeze the final model.
- Run the final test once.
- Confirm the demo works from a clean start.
- Prepare the explanation of robustness, unseen-generator generalization, false positives, and limitations.

## Hard fallback

If time becomes tight, submit:

- Whole-image pretrained ResNet-50
- Robust probabilistic augmentation
- Generator-disjoint testing
- Transformation-wise evaluation
- Working demo

Patching and greedy soup are improvements, not prerequisites for a credible prototype.