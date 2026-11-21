# Image Enhancement & Color Generation (CGIP Mini Project)

An interactive **Streamlit** and **OpenCV** web application for real-time digital image enhancement and color/pseudo-color generation. Designed as a Computer Graphics and Image Processing (CGIP) mini project, it lets users upload images, snap photos via webcam, or test with a built-in synthetic pattern while adjusting enhancement and color parameters live.

---

## Features

- **Flexible Input Sources**
  - **Upload File (`JPG`, `JPEG`, `PNG`)**: Upload images from your device.
  - **Built-in Test Pattern**: Instant gradient + geometric shape fallback image (`make_sample_image()`) so you can test immediately without preparing files.
  - **Camera Snapshot**: Capture an image using your webcam via `st.camera_input`.
- **Image Enhancement Pipeline**
  - **CLAHE (`LAB` Lightness Channel)**: Contrast Limited Adaptive Histogram Equalization (`clipLimit` adjustable from `1.0` to `6.0`, `8x8` tile grid) applied on the `L` channel to boost local contrast while preserving color balance.
  - **Gamma Correction**: Power-law intensity transformation (`0.2` to `3.0`) implemented using a 256-entry lookup table (`cv2.LUT`).
  - **Unsharp Masking (Sharpening)**: Edge enhancement (`0.0` to `2.5` strength) using Gaussian blur (`sigmaX=1.8`) and weighted blending (`cv2.addWeighted`).
- **Color & False-Color Generation**
  - **HSV Manipulation**: Adjust **Hue Shift** (`0`–`179`) and **Saturation Multiplier** (`0.0`–`2.5`).
  - **Pseudo-Color / False-Color Colormaps**: Convert intensity to false-color palettes (`Turbo`, `Inferno`, `Ocean`, `Bone`) via `cv2.applyColorMap`.
- **Side-by-Side Comparison & Export**
  - Displays original and processed images side by side with resolution and active color mode metadata, plus a **Download Processed Image (PNG)** button.

---

## Project Structure

```text
cgip-image-enhancer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## Installation

1. **Clone or create your project folder** and place your Streamlit script in `app.py`.

2. **Create a `requirements.txt` file** with:
   ```text
   streamlit
   opencv-python
   pillow
   numpy
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

Start the Streamlit server from your project directory:

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## Processing Order

The pipeline executes sequentially on a copy of the input image (`BGR` format):

1. **CLAHE** (if enabled) in `CIELAB` space on the `L` channel → converted back to `BGR`
2. **Gamma Correction** via lookup table
3. **Sharpening** via unsharp masking
4. **Color Generation** (HSV adjustment, followed by grayscale colormap mapping if a false-color mode is selected)
5. **Display & Export** (`BGR` converted to `RGB` for Streamlit rendering; encoded as `.png` for download)

---

## Tech Stack

- **Python**
- **Streamlit** (interactive UI, sidebar controls, camera input, download button)
- **OpenCV (`cv2`)** (color space conversions, CLAHE, LUT gamma, Gaussian blur, colormaps, PNG encoding)
- **NumPy** (array manipulation and lookup tables)
- **Pillow (`PIL`)** (image loading and RGB normalization)