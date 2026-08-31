"""Interactive dashboard for the AIGC image detector.
 
Loads a checkpoint produced by train.py (via the same `build_model` used
there), lets you upload an image, apply the real-world transform grid from
PLAN.md live, and watch the prediction survive (or fail) as you crank up
severity. Also runs a full degradation sweep across all transforms so the
robustness story is visible as a curve, not just a single anecdote.
 
Run with:
    streamlit run app.py -- --checkpoint checkpoints/best.pt
 
Assumption (flag if wrong): label 1 = AI-generated, label 0 = real. This
matches the convention implied by `pos_weight` in train.py's
BCEWithLogitsLoss. If your dataset labels it the other way round, flip
`AI_LABEL` below.
"""
 
from __future__ import annotations
 
import io
import sys
import argparse
from dataclasses import dataclass, field
 
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms as T
 
# build_model must match the architecture/weights layout used in train.py
# (a torchvision resnet50 with a replaced `.fc` head, per train.py's
# `model.fc.parameters()` usage).
from evaluate import build_model
 
AI_LABEL = 1  # 1 = AI-generated, 0 = real. See module docstring.
 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224
 
 
# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
 
@st.cache_resource(show_spinner="Loading model...")
def load_model(checkpoint_path: str, device_str: str):
    device = torch.device(device_str)
    model = build_model(num_classes=1, pretrained=False).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    meta = {k: v for k, v in state.items() if k != "model"}
    return model, device, meta
 
 
def preprocess(img: Image.Image) -> torch.Tensor:
    tf = T.Compose(
        [
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return tf(img.convert("RGB")).unsqueeze(0)
 
 
@torch.no_grad()
def predict(model, device, img: Image.Image) -> float:
    """Returns P(AI-generated) in [0, 1]."""
    x = preprocess(img).to(device)
    logit = model(x)
    prob = torch.sigmoid(logit).item()
    return prob if AI_LABEL == 1 else 1.0 - prob
 
 
# --------------------------------------------------------------------------
# Transform grid (matches PLAN.md exactly)
# --------------------------------------------------------------------------
 
@dataclass
class TransformParams:
    jpeg_quality: int = 100          # 100 = off; table values: 90/70/50/30
    blur_sigma: float = 0.0          # table values: 0.5/1.0/2.0
    resize_scale: float = 1.0        # table values: 0.5/0.25
    noise_sigma: float = 0.0         # table values: 0.02/0.05/0.10
    brightness: float = 0.0          # +/- 0.20
    contrast: float = 0.0            # +/- 0.20
    saturation: float = 0.0          # +/- 0.20
    crop_pct: float = 100.0          # table value: 80
 
 
def apply_transforms(img: Image.Image, p: TransformParams) -> Image.Image:
    """Apply the full transform grid in a realistic redistribution order:
    crop -> resize (thumbnail then upscale) -> blur -> color jitter ->
    noise -> JPEG re-encode (JPEG last, since re-encoding is typically the
    final step when an image is redistributed).
    """
    out = img.convert("RGB")
    w, h = out.size
 
    # Center crop
    if p.crop_pct < 100.0:
        frac = p.crop_pct / 100.0
        cw, ch = int(w * frac), int(h * frac)
        left, top = (w - cw) // 2, (h - ch) // 2
        out = out.crop((left, top, left + cw, top + ch))
 
    # Downscale then upscale (thumbnail-and-back simulates real degradation)
    if p.resize_scale < 1.0:
        ow, oh = out.size
        small = out.resize(
            (max(1, int(ow * p.resize_scale)), max(1, int(oh * p.resize_scale))),
            Image.BICUBIC,
        )
        out = small.resize((ow, oh), Image.BICUBIC)
 
    # Gaussian blur
    if p.blur_sigma > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=p.blur_sigma))
 
    # Color jitter: brightness / contrast / saturation, each as a +/- factor
    if p.brightness != 0.0:
        out = ImageEnhance.Brightness(out).enhance(1.0 + p.brightness)
    if p.contrast != 0.0:
        out = ImageEnhance.Contrast(out).enhance(1.0 + p.contrast)
    if p.saturation != 0.0:
        out = ImageEnhance.Color(out).enhance(1.0 + p.saturation)
 
    # Gaussian noise (sigma in [0,1] pixel-intensity units)
    if p.noise_sigma > 0:
        arr = np.asarray(out).astype(np.float32) / 255.0
        noise = np.random.normal(0.0, p.noise_sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        out = Image.fromarray((arr * 255).astype(np.uint8))
 
    # JPEG re-encode
    if p.jpeg_quality < 100:
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=p.jpeg_quality)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
 
    return out
 
 
# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
 
def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Streamlit passes its own args through sys.argv before `--`; only parse
    # what comes after `--` if present, otherwise fall back to defaults.
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []
    return parser.parse_args(argv)
 
 
def confidence_bar(prob_ai: float):
    label = "AI-GENERATED" if prob_ai >= 0.5 else "REAL"
    color = "#d9534f" if prob_ai >= 0.5 else "#5cb85c"
    conf = prob_ai if prob_ai >= 0.5 else 1.0 - prob_ai
    st.markdown(
        f"""
        <div style="padding:12px;border-radius:8px;background:#1e1e1e;border:1px solid {color};">
          <span style="font-size:1.3em;font-weight:700;color:{color};">{label}</span>
          <span style="float:right;font-size:1.3em;color:{color};">{conf*100:.1f}%</span>
          <div style="margin-top:8px;height:10px;border-radius:5px;background:#333;overflow:hidden;">
            <div style="width:{conf*100:.1f}%;height:100%;background:{color};"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
 
SWEEP_CONFIG = {
    "JPEG quality": ("jpeg_quality", [100, 90, 70, 50, 30]),
    "Gaussian blur σ": ("blur_sigma", [0.0, 0.5, 1.0, 2.0]),
    "Resize scale": ("resize_scale", [1.0, 0.5, 0.25]),
    "Gaussian noise σ": ("noise_sigma", [0.0, 0.02, 0.05, 0.10]),
    "Center crop %": ("crop_pct", [100.0, 80.0]),
}
 
 
def run_sweep(model, device, base_img: Image.Image):
    """For each transform dimension, hold others at baseline and sweep
    severity, recording P(AI). Returns {transform_name: (severities, probs)}.
    """
    results = {}
    for name, (field_name, values) in SWEEP_CONFIG.items():
        probs = []
        for v in values:
            params = TransformParams(**{field_name: v})
            timg = apply_transforms(base_img, params)
            probs.append(predict(model, device, timg))
        results[name] = (values, probs)
    return results
 
 
def main():
    st.set_page_config(page_title="AIGC Detector", layout="wide")
    st.title("AI-Generated Image Detector")
    st.caption(
        "Upload an image, apply real-world transformations, and watch whether "
        "the prediction survives."
    )
 
    args = parse_cli_args()
 
    with st.sidebar:
        st.header("Model")
        checkpoint_path = st.text_input("Checkpoint path", value=args.checkpoint)
        device_str = st.selectbox(
            "Device", options=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
            index=0,
        )
        try:
            model, device, meta = load_model(checkpoint_path, device_str)
            st.success(f"Loaded checkpoint (epoch {meta.get('epoch', '?')}, "
                       f"score {meta.get('selection_score', float('nan')):.4f})")
        except Exception as e:
            st.error(f"Could not load checkpoint: {e}")
            st.stop()
 
        st.header("Transforms")
        st.caption("Set to the leftmost/off value to see the clean prediction.")
        jpeg_quality = st.select_slider("JPEG quality", options=[100, 90, 70, 50, 30], value=100)
        blur_sigma = st.select_slider("Gaussian blur σ", options=[0.0, 0.5, 1.0, 2.0], value=0.0)
        resize_scale = st.select_slider("Resize scale", options=[1.0, 0.5, 0.25], value=1.0)
        noise_sigma = st.select_slider("Gaussian noise σ", options=[0.0, 0.02, 0.05, 0.10], value=0.0)
        brightness = st.slider("Brightness Δ", -0.20, 0.20, 0.0, 0.05)
        contrast = st.slider("Contrast Δ", -0.20, 0.20, 0.0, 0.05)
        saturation = st.slider("Saturation Δ", -0.20, 0.20, 0.0, 0.05)
        crop_pct = st.select_slider("Center crop %", options=[100.0, 80.0], value=100.0)
 
    params = TransformParams(
        jpeg_quality=jpeg_quality,
        blur_sigma=blur_sigma,
        resize_scale=resize_scale,
        noise_sigma=noise_sigma,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        crop_pct=crop_pct,
    )
 
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is None:
        st.info("Upload an image to run inference, or use the sweep below on a sample.")
        return
 
    orig_img = Image.open(uploaded).convert("RGB")
    trans_img = apply_transforms(orig_img, params)
 
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(orig_img, use_container_width=True)
        prob_clean = predict(model, device, orig_img)
        confidence_bar(prob_clean)
 
    with col2:
        st.subheader("Transformed")
        st.image(trans_img, use_container_width=True)
        prob_trans = predict(model, device, trans_img)
        confidence_bar(prob_trans)
 
    delta = abs(prob_trans - prob_clean)
    if delta > 0.15:
        st.warning(
            f"⚠️ Prediction shifted by {delta*100:.1f} points under this transform"
        )
    else:
        st.success(f"✅ Prediction stable (Δ = {delta*100:.1f} points) under this transform.")
 
    st.divider()
    st.subheader("Degradation sweep")
    st.caption(
        "Holds all other transforms at baseline and sweeps one dimension at a time, "
        "using the image above."
    )
    if st.button("Run full sweep"):
        with st.spinner("Running inference across the transform grid..."):
            sweep = run_sweep(model, device, orig_img)
        sweep_cols = st.columns(len(sweep))
        for (name, (severities, probs)), col in zip(sweep.items(), sweep_cols):
            with col:
                st.markdown(f"**{name}**")
                st.line_chart(
                    {"P(AI-generated)": probs},
                    x=[str(v) for v in severities],
                    height=200,
                )
 
 
if __name__ == "__main__":
    main()
