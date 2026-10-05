"""
Smart Image Quality Enhancement and Object Segmentation
-------------------------------------------------------
Application : Low-light object image
Run         : streamlit run app.py
 
Workflow:
Input -> Pre-processing (grayscale) -> Noise removal/Smoothing -> Histogram analysis
      -> Contrast enhancement -> Thresholding -> Morphology -> Connected components -> Final output
"""
import os
 
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # no pop-up windows, needed for Streamlit
import matplotlib.pyplot as plt
import streamlit as st
 
st.set_page_config(page_title="Smart Image Segmentation", layout="wide")
 
SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "sample_images", "sample_low_light.jpg")
MAX_SIZE = 900                             # large images are shrunk so the app stays fast
 
 
# ----------------------------------------------------------------------------
# 1. Helper functions (each one is a stage of the pipeline)
# ----------------------------------------------------------------------------
def load_image(data):
    """Decode bytes into a 3-channel BGR image. Handles gray, RGBA and 16-bit images.
    Returns (bgr_image, original_channel_count) or (None, 0) if the file is not an image."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, 0
    if img.dtype != np.uint8:              # e.g. 16-bit PNG
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    channels = 1 if img.ndim == 2 else img.shape[2]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif channels == 4:                    # drop transparency channel
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIZE:
        s = MAX_SIZE / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img, channels
 
 
def remove_noise(gray, method, k):
    """Gaussian blur suits sensor (Gaussian) noise; median blur suits salt-and-pepper noise."""
    if method == "Gaussian Blur":
        return cv2.GaussianBlur(gray, (k, k), 0)
    if method == "Median Blur":
        return cv2.medianBlur(gray, k)
    return gray.copy()
 
 
def enhance(gray, method, clip):
    """CLAHE = histogram equalization on small tiles with a clip limit (less noise boost)."""
    if method == "CLAHE":
        return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray)
    if method == "Histogram Equalization":
        return cv2.equalizeHist(gray)
    return gray.copy()
 
 
def segment(enh, method, thr_value, block, c):
    """Return (binary_image, threshold_used). threshold_used is None for adaptive."""
    if method == "Global Threshold":
        t, b = cv2.threshold(enh, thr_value, 255, cv2.THRESH_BINARY)
        return b, float(t)
    if method == "Otsu Threshold":
        t, b = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return b, float(t)
    # Adaptive: each pixel is compared with the mean of its neighbourhood.
    # Passing -c means "pixel must be brighter than local mean + c" (bright object).
    b = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, block, -c)
    return b, None
 
 
def fix_polarity(binary, mode):
    """Make the object WHITE. 'Auto' assumes the image border is background."""
    if mode == "Bright object on dark background":
        return binary
    if mode == "Dark object on bright background":
        return 255 - binary
    border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    return 255 - binary if border.mean() > 127 else binary
 
 
def apply_morphology(mask, ops, k):
    """Opening removes small white specks, closing fills small black gaps."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    table = {
        "Opening": lambda m: cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel),
        "Closing": lambda m: cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel),
        "Dilation": lambda m: cv2.dilate(m, kernel),
        "Erosion": lambda m: cv2.erode(m, kernel),
    }
    for name in ["Opening", "Closing", "Dilation", "Erosion"]:   # fixed, predictable order
        if name in ops:
            mask = table[name](mask)
    return mask
 
 
def pick_object(mask, fill_holes):
    """Connected components (8-connectivity): keep the largest region as the object."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask), labels, n - 1, None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    obj = np.where(labels == idx, 255, 0).astype(np.uint8)
    if fill_holes and obj[0, 0] == 0:          # fill black holes inside the object
        inv = 255 - obj
        ff = inv.copy()
        cv2.floodFill(ff, np.zeros((obj.shape[0] + 2, obj.shape[1] + 2), np.uint8), (0, 0), 0)
        obj = cv2.bitwise_or(obj, ff)
    return obj, labels, n - 1, idx
 
 
def colour_components(labels, n_regions, chosen):
    """Colour every connected region differently; the chosen object is green."""
    rng = np.random.RandomState(3)
    colours = rng.randint(70, 255, (n_regions + 1, 3)).astype(np.uint8)
    colours[0] = 0
    if chosen is not None:
        colours[chosen] = (0, 255, 0)
    return colours[labels]
 
 
def histogram_figure(gray, title):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.hist(gray.ravel(), bins=256, range=(0, 255), color="#4a6cf7")
    ax.set_title(title)
    ax.set_xlabel("Pixel intensity (0-255)")
    ax.set_ylabel("Pixel count")
    fig.tight_layout()
    return fig
 
 
def show(img, caption):
    """Display an image (BGR or gray) correctly in Streamlit."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        st.image(img, caption=caption, width="stretch")          # newer Streamlit
    except Exception:
        st.image(img, caption=caption, use_container_width=True)  # older Streamlit
 
 
# ----------------------------------------------------------------------------
# 2. Sidebar controls
# ----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("processed", False)
ss.setdefault("use_sample", False)
ss.setdefault("uploader_key", 0)
 
st.sidebar.title("Controls")
uploaded = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"],
                                    key=f"up{ss.uploader_key}")
if st.sidebar.button("Use sample low-light image"):
    ss.use_sample, ss.processed = True, True
 
denoise_method = st.sidebar.selectbox("Blur / Noise Removal Method",
                                      ["Gaussian Blur", "Median Blur", "None"])
blur_k = st.sidebar.slider("Blur kernel size", 3, 11, 5, step=2)
enh_method = st.sidebar.selectbox("Enhancement Method",
                                  ["CLAHE", "Histogram Equalization", "None"])
clip = st.sidebar.slider("CLAHE clip limit", 1.0, 8.0, 3.0, step=0.5)
seg_method = st.sidebar.selectbox("Segmentation Method",
                                  ["Otsu Threshold", "Global Threshold", "Adaptive Threshold"])
g_thr = st.sidebar.slider("Global threshold value", 0, 255, 40)
a_block = st.sidebar.slider("Adaptive block size", 11, 101, 51, step=2)
a_c = st.sidebar.slider("Adaptive constant C", 0, 30, 5)
polarity = st.sidebar.selectbox("Object type", ["Auto (border = background)",
                                                "Bright object on dark background",
                                                "Dark object on bright background"])
morph_ops = st.sidebar.multiselect("Morphological Operation",
                                   ["Opening", "Closing", "Dilation", "Erosion"],
                                   default=["Opening", "Closing"])
morph_k = st.sidebar.slider("Morphology kernel size", 3, 21, 7, step=2)
fill = st.sidebar.checkbox("Fill holes inside the object", value=True)
 
c1, c2 = st.sidebar.columns(2)
if c1.button("Process Image"):
    ss.processed = True
if c2.button("Reset"):
    ss.processed, ss.use_sample = False, False
    ss.uploader_key += 1
    st.rerun()
 
# ----------------------------------------------------------------------------
# 3. Main page
# ----------------------------------------------------------------------------
st.title("Smart Image Quality Enhancement and Object Segmentation")
st.caption("Low-light object image application | Python + OpenCV + Streamlit")
 
with st.expander("Concepts used (simple explanations for viva)"):
    st.markdown("""
- **Histogram** - a bar graph of how many pixels have each brightness (0 = black, 255 = white). A low-light image has all its pixels bunched near 0.
- **Histogram Equalization / CLAHE** - spreads the brightness values out to raise contrast. CLAHE does it on small tiles and limits the boost, so noise is not over-amplified.
- **Noise Removal** - low-light photos are grainy. Median blur removes salt-and-pepper dots, Gaussian blur reduces random sensor noise.
- **Smoothing** - averaging each pixel with its neighbours; it softens the image so thresholding does not pick up grain.
- **Thresholding** - pixel > T becomes white (object), otherwise black (background).
- **Otsu Thresholding** - finds the best single T automatically by separating the histogram into two groups.
- **Adaptive Thresholding** - uses a different T for every small neighbourhood, useful when lighting is uneven.
- **Morphological Operations** - erosion shrinks white regions, dilation grows them; opening (erode then dilate) removes specks, closing (dilate then erode) fills gaps.
- **Connectivity** - pixels touching each other (8 neighbours) form one connected component; we keep the largest one as the object.
""")
 
# ---- Section 1: input -------------------------------------------------------
st.header("Section 1: Input Image")
data, source_name = None, ""
if uploaded is not None:
    data, source_name = uploaded.getvalue(), uploaded.name
elif ss.use_sample:
    if os.path.exists(SAMPLE_PATH):
        with open(SAMPLE_PATH, "rb") as f:
            data, source_name = f.read(), "sample_low_light.jpg"
    else:
        st.error("Sample image not found. Run make_sample_image.py first.")
 
if data is None:
    st.info("Upload a JPG/PNG in the sidebar (or use the sample), then click **Process Image**.")
    st.stop()
 
img, channels = load_image(data)
if img is None:
    st.error("This file could not be read as an image. Please upload a valid JPG, JPEG or PNG.")
    st.stop()
 
h, w = img.shape[:2]
col_a, col_b = st.columns([2, 1])
with col_a:
    show(img, f"Original image: {source_name}")
with col_b:
    st.write(f"**Width:** {w} px")
    st.write(f"**Height:** {h} px")
    st.write(f"**Channels in file:** {channels}")
if not ss.processed:
    st.info("Click **Process Image** in the sidebar to run the pipeline.")
    st.stop()
 
# ---- Run the pipeline -------------------------------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smooth = remove_noise(gray, denoise_method, blur_k)
enh = enhance(smooth, enh_method, clip)
binary, thr_used = segment(enh, seg_method, g_thr, a_block, a_c)
binary = fix_polarity(binary, polarity)
morph = apply_morphology(binary, morph_ops, morph_k)
obj_mask, labels, n_regions, chosen = pick_object(morph, fill)
cc_vis = colour_components(labels, max(n_regions, 0), chosen)
cutout = cv2.bitwise_and(img, img, mask=obj_mask)                  # object on black
overlay = img.copy()
contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
 
# ---- Section 2: pre-processing ---------------------------------------------
st.header("Section 2: Pre-processing")
a, b = st.columns(2)
with a:
    show(gray, "Grayscale image")
with b:
    show(smooth, f"Noise-reduced image ({denoise_method}, kernel {blur_k})")
st.write("Gray conversion removes colour so only brightness is processed. "
         "Noise removal comes before enhancement because contrast enhancement also boosts noise.")
 
# ---- Section 3: histogram analysis -----------------------------------------
st.header("Section 3: Histogram Analysis")
fig = histogram_figure(smooth, "Histogram before enhancement")
st.pyplot(fig)
plt.close(fig)
st.write("If most bars sit at the left (dark values), the image is under-exposed.")
 
# ---- Section 4: enhancement --------------------------------------------------
st.header("Section 4: Contrast Enhancement")
a, b = st.columns(2)
with a:
    show(smooth, "Before enhancement")
    fig = histogram_figure(smooth, "Histogram before")
    st.pyplot(fig)
    plt.close(fig)
with b:
    show(enh, f"After enhancement ({enh_method})")
    fig = histogram_figure(enh, "Histogram after")
    st.pyplot(fig)
    plt.close(fig)
 
# ---- Section 5: thresholding ------------------------------------------------
st.header("Section 5: Thresholding / Segmentation")
show(binary, f"Binary result ({seg_method}), object shown in white")
if thr_used is not None:
    st.write(f"Threshold value used: **{thr_used:.0f}**")
else:
    st.write("Adaptive thresholding uses a different threshold for every pixel (no single value).")
 
# ---- Section 6: morphology --------------------------------------------------
st.header("Section 6: Morphological Processing")
a, b = st.columns(2)
with a:
    show(binary, "Before morphology")
with b:
    show(morph, f"After morphology ({', '.join(morph_ops) or 'none'}, kernel {morph_k})")
 
st.subheader("Connected components")
a, b = st.columns(2)
with a:
    show(cc_vis, f"{max(n_regions, 0)} connected regions found (chosen object = green)")
with b:
    show(obj_mask, "Mask of the largest region")
 
# ---- Section 7: final object -------------------------------------------------
st.header("Section 7: Final Segmented Object")
if chosen is None:
    st.warning("No object was detected. Try another segmentation method, change the "
               "object type, or lower the morphology kernel size (see README troubleshooting).")
a, b = st.columns(2)
with a:
    show(cutout, "Segmented object (background removed)")
with b:
    show(overlay, "Object outline on original")
 
# ---- Section 8: comparison ---------------------------------------------------
st.header("Section 8: Comparison of Results")
cols = st.columns(4)
steps = [(img, "Original", "Dark, noisy input image."),
         (enh, "Enhanced", "Noise reduced, contrast stretched."),
         (binary, "Thresholded", "Pixels split into object / background."),
         (cutout, "Final Segmented", "Largest connected region kept, background removed.")]
for col, (im, title, text) in zip(cols, steps):
    with col:
        show(im, title)
        st.caption(text)
 
# ---- Section 9: metrics ------------------------------------------------------
st.header("Section 9: Image Metrics")
fg = int(np.count_nonzero(obj_mask))
total = obj_mask.size
m = st.columns(3)
m[0].metric("Original mean intensity", f"{gray.mean():.2f}")
m[1].metric("Enhanced mean intensity", f"{enh.mean():.2f}")
m[2].metric("Threshold value", f"{thr_used:.0f}" if thr_used is not None else "Adaptive")
m = st.columns(3)
m[0].metric("Foreground area (px)", f"{fg}")
m[1].metric("Background area (px)", f"{total - fg}")
m[2].metric("Foreground %", f"{100 * fg / total:.2f}%")
st.caption("All values are calculated from the processed image; nothing is hard-coded.")