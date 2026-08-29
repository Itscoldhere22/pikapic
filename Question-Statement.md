5. Robust Detection of AI Generated Images Under Real World Transformations

# 5.1 Problem Statement
We want participants to build a prototype that can distinguish AI-generated images from authentic images with strong robustness under realistic post-processing and redistribution scenarios. The goal is not only to achieve good detection performance on clean data, but also to maintain accuracy after transformations such as blur, compression, color adjustment, cropping, or rescaling. Solutions should present a clear technical approach, an evaluation strategy, and thoughtful discussion of trade-offs such as robustness, generalisation, and false positives.
**Note: We consider robustness against a subset of the following augmentataions.**
| Transform | Parameters | Real-World Analog |
|---|---|---|
| JPEG Compression | quality = 90, 70, 50, 30 | Social-media re-encode, messaging |
| Gaussian Blur | kernel σ = 0.5, 1.0, 2.0 | Out-of-focus |
| Resize | scale 0.5x / 0.25x then upscale | Thumbnail generation |
| Gaussian Noise | σ = 0.02, 0.05, 0.10 | Low-light sensor noise |
| Color Jitter | brightness/contrast/sat. ±20% | Filter apps, auto-enhance |
| Center Crop | crop 80% | Profile-picture cropping, framing |

# 5.3 Constraints & Scope
| Category | Constraints & Scope Details |
|---|---|
| In scope | Image-level AIGC detection, robustness to common image transformations, feature engineering, model design, evaluation design, error analysis, and explainability ideas |
| Out of scope | Full production deployment, platform-wide moderation systems, and non-image modalities such as video or audio |
| Limits | Assume a hackathon-scale prototype, limited compute, and no access to internal production systems. Teams should optimise for a convincing proof of concept rather than a production-grade service. Note: Participants must use models with <2B parameters. |
| Allowed assumptions | Teams may use public or properly licensed datasets, create their own transformed test cases, and make reasonable assumptions about deployment context as long as those assumptions are stated clearly. |