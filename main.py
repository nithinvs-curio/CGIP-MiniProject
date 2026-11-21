import cv2
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="CGIP Image Enhancer & Color Generator",
    layout="wide",
)

st.title("Image Enhancement & Color Generation (CGIP Mini Project)")
st.caption("Upload an image or use the built-in test pattern on the left to experiment in real time.")


def make_sample_image() -> np.ndarray:
    """Fallback gradient + geometric shape image so you can test immediately."""
    x = np.linspace(0, 255, 400, dtype=np.uint8)
    grid = np.tile(x, (300, 1))
    img = cv2.merge([grid, np.roll(grid, 80, axis=1), np.roll(grid, 160, axis=1)])
    cv2.circle(img, (120, 150), 55, (240, 240, 240), -1)
    cv2.rectangle(img, (210, 95), (340, 205), (40, 40, 40), -1)
    return img


def apply_clahe_lab(img_bgr: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on LAB Lightness channel."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_enhanced = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge([l_enhanced, a_chan, b_chan]), cv2.COLOR_LAB2BGR)


def apply_gamma(img_bgr: np.ndarray, gamma: float) -> np.ndarray:
    """Power-law (gamma) intensity transformation using a 256-entry lookup table."""
    inv_gamma = 1.0 / max(gamma, 0.01)
    lut = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img_bgr, lut)


def apply_sharpen(img_bgr: np.ndarray, amount: float, sigma: float = 1.8) -> np.ndarray:
    """Unsharp masking for edge sharpening."""
    if amount <= 0:
        return img_bgr
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(img_bgr, 1.0 + amount, blurred, -amount, 0)


def apply_color_mode(
    img_bgr: np.ndarray,
    mode: str,
    hue_shift: int,
    sat_scale: float,
) -> np.ndarray:
    """Adjust HSV channels and optionally map grayscale intensity to a pseudo-color map."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0.0, 255.0)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    colormaps = {
        "Original / HSV Adjusted": None,
        "False Color - Turbo": cv2.COLORMAP_TURBO,
        "False Color - Inferno": cv2.COLORMAP_INFERNO,
        "False Color - Ocean": cv2.COLORMAP_OCEAN,
        "False Color - Bone": cv2.COLORMAP_BONE,
    }
    cmap = colormaps.get(mode)
    if cmap is not None:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.applyColorMap(gray, cmap)
    return out


# ---------------- Sidebar: Input & Controls ----------------
st.sidebar.header("1. Input Image")
input_source = st.sidebar.radio(
    "Choose input source",
    ["Upload File (JPG/PNG)", "Built-in Test Pattern", "Camera Snapshot"],
)

original_bgr = None

if input_source == "Upload File (JPG/PNG)":
    uploaded = st.sidebar.file_uploader("Upload JPG / PNG", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        original_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        st.sidebar.info("No file uploaded yet — showing built-in test pattern.")
        original_bgr = make_sample_image()

elif input_source == "Camera Snapshot":
    camera_photo = st.camera_input("Take a photo")
    if camera_photo is not None:
        pil_img = Image.open(camera_photo).convert("RGB")
        original_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        st.info("Capture a photo above or switch input source in the sidebar.")
        original_bgr = make_sample_image()

else:
    original_bgr = make_sample_image()

st.sidebar.header("2. Image Enhancement")
use_clahe = st.sidebar.checkbox("Apply CLAHE (Contrast Limited)", value=False)
clip_limit = st.sidebar.slider("CLAHE Clip Limit", 1.0, 6.0, 2.0, 0.2, disabled=not use_clahe)
gamma = st.sidebar.slider("Gamma Correction", 0.2, 3.0, 1.0, 0.05)
sharpen = st.sidebar.slider("Sharpening Strength", 0.0, 2.5, 0.0, 0.1)

st.sidebar.header("3. Color Generation")
color_mode = st.sidebar.selectbox(
    "Color Mode",
    [
        "Original / HSV Adjusted",
        "False Color - Turbo",
        "False Color - Inferno",
        "False Color - Ocean",
        "False Color - Bone",
    ],
)
hue_shift = st.sidebar.slider("Hue Shift (HSV)", 0, 179, 0, 1)
sat_scale = st.sidebar.slider("Saturation Multiplier", 0.0, 2.5, 1.0, 0.05)

# ---------------- Processing Pipeline ----------------
processed = original_bgr.copy()

if use_clahe:
    processed = apply_clahe_lab(processed, clip_limit=clip_limit, grid_size=8)

processed = apply_gamma(processed, gamma)
processed = apply_sharpen(processed, sharpen)
processed = apply_color_mode(processed, color_mode, hue_shift, sat_scale)

# ---------------- Display ----------------
orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
proc_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

h, w = original_bgr.shape[:2]
st.markdown(f"**Resolution:** `{w} × {h}` px | **Mode:** `{color_mode}`")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Original Image")
    st.image(orig_rgb, use_container_width=True)

with col2:
    st.subheader("Processed Output")
    st.image(proc_rgb, use_container_width=True)

# Download button
success, buf = cv2.imencode(".png", processed)
if success:
    st.download_button(
        label="Download Processed Image (PNG)",
        data=buf.tobytes(),
        file_name="cgip_output.png",
        mime="image/png",
    )