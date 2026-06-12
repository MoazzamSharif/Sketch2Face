import streamlit as st
import streamlit.components.v1 as components
import subprocess
import os
import shutil
import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from io import BytesIO
import base64
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="Sketch to Face", page_icon="✏", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif; 
}

.stApp { 
    background: radial-gradient(circle at 10% 20%, #faf8f3 0%, #f7f3eb 100%); 
    color: #4a433d; 
}

.block-container { 
    max-width: 820px; 
    padding-top: 3.5rem; 
    padding-bottom: 5rem; 
}

.site-title { 
    font-family: 'Outfit', sans-serif;
    font-size: 2.5rem; 
    font-weight: 800; 
    letter-spacing: -0.02em; 
    color: #2b2520; 
    margin: 0 0 0.3rem 0; 
}

.site-subtitle { 
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem; 
    color: #8d8378; 
    font-weight: 400; 
    margin: 0 0 2.5rem 0; 
    line-height: 1.6; 
}

[data-testid="stFileUploader"] { 
    background: #ffffff; 
    border: 2px dashed #e6ddd2; 
    border-radius: 16px; 
    padding: 2rem; 
    transition: all 0.3s cubic-bezier(0.25, 1, 0.5, 1); 
    box-shadow: 0 4px 20px rgba(74, 67, 61, 0.03); 
}

[data-testid="stFileUploader"]:hover { 
    border-color: #b38a5d; 
    box-shadow: 0 8px 30px rgba(214, 184, 146, 0.15); 
}

[data-testid="stFileUploadDropzone"] p { 
    color: #8d8378 !important; 
    font-size: 0.9rem !important; 
}

[data-testid="stImage"] img { 
    border-radius: 14px; 
    width: 100%; 
    display: block; 
    border: 1px solid #ece5dc; 
    box-shadow: 0 4px 15px rgba(74, 67, 61, 0.04);
    transition: transform 0.3s ease;
}

[data-testid="stImage"] img:hover {
    transform: scale(1.015);
}

.img-label { 
    font-family: 'Outfit', sans-serif;
    font-size: 0.72rem; 
    color: #b38a5d; 
    text-align: center; 
    margin-top: 8px; 
    letter-spacing: 0.08em; 
    text-transform: uppercase; 
    font-weight: 600;
}

.arrow { 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    color: #c0b2a2; 
    font-size: 1.6rem; 
    height: 100%; 
    padding-top: 80px; 
    text-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

[data-testid="stTabs"] [data-baseweb="tab-list"] { 
    background: transparent; 
    border-bottom: 2px solid #e6ddd2; 
    gap: 8px; 
    margin-bottom: 2rem; 
}

[data-testid="stTabs"] [data-baseweb="tab"] { 
    background: transparent !important; 
    color: #9b8f82 !important; 
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.88rem !important; 
    font-weight: 600 !important; 
    padding: 0.7rem 1.5rem !important; 
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease !important;
}

[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    background: rgba(214, 184, 146, 0.04) !important;
    color: #b38a5d !important;
}

[data-testid="stTabs"] [aria-selected="true"] { 
    color: #b38a5d !important; 
    border-bottom: 2.5px solid #b38a5d !important; 
    background: rgba(214, 184, 146, 0.06) !important; 
}

.stButton > button[kind="primary"] { 
    background: linear-gradient(135deg, #d6b892 0%, #b38a5d 100%) !important; 
    color: white !important; 
    border: none !important; 
    border-radius: 12px !important; 
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important; 
    font-size: 0.95rem !important; 
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    height: 3rem !important; 
    box-shadow: 0 4px 15px rgba(214, 184, 146, 0.35) !important;
    transition: all 0.3s cubic-bezier(0.25, 1, 0.5, 1) !important; 
}

.stButton > button[kind="primary"]:hover { 
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(214, 184, 146, 0.5) !important;
    background: linear-gradient(135deg, #c9aa83 0%, #a07647 100%) !important;
}

.stButton > button[kind="primary"]:active {
    transform: translateY(0) !important;
}

.stDownloadButton > button { 
    background-color: white !important; 
    color: #7a7167 !important; 
    border: 1px solid #ddd3c8 !important; 
    border-radius: 10px !important; 
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.8rem !important; 
    font-weight: 600 !important; 
    height: 2.5rem !important; 
    box-shadow: 0 2px 8px rgba(74, 67, 61, 0.02) !important;
    transition: all 0.2s !important; 
}

.stDownloadButton > button:hover { 
    background-color: #faf8f3 !important; 
    color: #b38a5d !important; 
    border-color: #b38a5d !important; 
    box-shadow: 0 4px 12px rgba(214, 184, 146, 0.1) !important;
}

.stTextInput input, .stNumberInput input, .stTextArea textarea { 
    background-color: white !important; 
    color: #4a433d !important; 
    border: 1px solid #ddd3c8 !important; 
    border-radius: 10px !important; 
    padding: 0.4rem 0.8rem !important;
}

.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus { 
    border-color: #b38a5d !important; 
    box-shadow: 0 0 0 3px rgba(214, 184, 146, 0.18) !important; 
}

.stSelectbox div[data-baseweb="select"] > div { 
    background-color: white !important; 
    border: 1px solid #ddd3c8 !important; 
    color: #4a433d !important; 
    border-radius: 10px !important;
}

[data-testid="stSidebar"] { 
    background-color: #fcfaf6 !important; 
    border-right: 1px solid #e6ddd2 !important; 
}

[data-testid="stSidebar"] label { 
    color: #8d8378 !important; 
    font-size: 0.78rem !important; 
}

[data-testid="stAlert"] { 
    border-radius: 12px !important; 
    font-size: 0.88rem !important; 
    border: 1px solid #e2d8cc !important; 
    box-shadow: 0 4px 15px rgba(0,0,0,0.02) !important;
}

.stSpinner > div > div { 
    border-top-color: #b38a5d !important; 
}

hr { 
    border-color: #e6ddd2 !important; 
    margin: 2.5rem 0 !important; 
}

[data-testid="metric-container"] { 
    background: white; 
    border: 1px solid #ece5dc; 
    border-radius: 14px; 
    padding: 14px; 
    box-shadow: 0 4px 15px rgba(74, 67, 61, 0.02);
}

.consistency-card { 
    background: white; 
    border: 1px solid #ece5dc; 
    border-left: 4px solid #b38a5d;
    border-radius: 16px; 
    padding: 1.4rem 1.8rem; 
    margin-top: 1.2rem; 
    box-shadow: 0 4px 20px rgba(74, 67, 61, 0.04);
}

.consistency-title { 
    font-family: 'Outfit', sans-serif;
    font-size: 0.75rem; 
    color: #9a8f82; 
    letter-spacing: 0.08em; 
    text-transform: uppercase; 
    margin-bottom: 0.6rem; 
    font-weight: 700;
}

.consistency-score { 
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem; 
    font-weight: 800; 
    color: #2b2520; 
    letter-spacing: -0.02em; 
}

.consistency-verdict { 
    font-size: 0.85rem; 
    color: #8d8378; 
    margin-top: 0.3rem; 
    line-height: 1.5;
}

.section-header { 
    font-family: 'Outfit', sans-serif;
    font-size: 0.78rem; 
    color: #b38a5d; 
    letter-spacing: 0.1em; 
    text-transform: uppercase; 
    margin: 2.2rem 0 1rem 0; 
    font-weight: 700;
}

.fidelity-label { 
    font-size: 0.78rem; 
    color: #8d8378; 
    margin-bottom: 0.3rem; 
    font-weight: 500; 
}

.diff-caption { 
    font-size: 0.78rem; 
    color: #8d8378; 
    text-align: center; 
    margin-top: 6px; 
}

#MainMenu, footer, header { 
    visibility: hidden; 
}
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────
PYTHON_EXE      = r"C:\ProgramData\anaconda3\envs\sketch2face\python.exe"
PIX2PIX_REPO    = r"C:\Users\rendering_lab-01\Downloads\pytorch-CycleGAN-and-pix2pix"
CHECKPOINTS     = r"C:\Users\rendering_lab-01\Downloads\pix2pix_checkpoints"
CODEFORMER_REPO = r"C:\Users\rendering_lab-01\Downloads\CodeFormer"
EXP_NAME        = "sketch2face_pix2pix"
FACE_UPSAMPLE   = True
BASE_TEMP       = r"C:\Users\rendering_lab-01\Downloads\streamlit_s2f_temp"
TEMP_TEST_ROOT  = os.path.join(BASE_TEMP, "input")
TEMP_VAL_DIR    = os.path.join(TEMP_TEST_ROOT, "val")
PIX2PIX_RESULTS = os.path.join(PIX2PIX_REPO, "results")
TEMP_CF_OUT     = os.path.join(BASE_TEMP, "cf_output")

# ── Helpers ───────────────────────────────────────────────────────
@st.cache_resource
def load_detector():
    from controlnet_aux import LineartDetector
    return LineartDetector.from_pretrained("lllyasviel/Annotators")

def to_png_bytes(arr):
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()

def img_to_b64(arr):
    return base64.b64encode(to_png_bytes(arr)).decode()

def crop_face(bgr, padding=0.4):
    gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces   = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    pad = int(max(w, h) * padding)
    x1, y1 = max(0, x-pad), max(0, y-pad)
    x2, y2 = min(bgr.shape[1], x+w+pad), min(bgr.shape[0], y+h+pad)
    crop = bgr[y1:y2, x1:x2]; h2, w2 = crop.shape[:2]; s = min(h2, w2)
    return crop[(h2-s)//2:(h2-s)//2+s, (w2-s)//2:(w2-s)//2+s]

def make_sketch(bgr_256):
    detector = load_detector()
    pil      = Image.fromarray(cv2.cvtColor(bgr_256, cv2.COLOR_BGR2RGB))
    lineart  = detector(pil, detect_resolution=256, image_resolution=256)
    lineart  = ImageOps.invert(lineart.convert("RGB"))
    return cv2.cvtColor(np.array(lineart), cv2.COLOR_RGB2BGR)

def save_pix2pix_input(sketch_bgr, filename):
    os.makedirs(TEMP_VAL_DIR, exist_ok=True)
    sketch_rgb = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2RGB)
    paired     = np.concatenate([sketch_rgb, np.zeros_like(sketch_rgb)], axis=1)
    Image.fromarray(paired).save(os.path.join(TEMP_VAL_DIR, filename))

def run_cmd(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

def find_pix2pix_output(stem):
    folder = Path(PIX2PIX_RESULTS) / EXP_NAME / "val_latest" / "images"
    direct = folder / f"{stem}_fake_B.png"
    if direct.exists(): return str(direct)
    matches = sorted(folder.glob("*_fake_B.png")) if folder.exists() else []
    return str(matches[-1]) if matches else None

def find_cf_output(stem):
    fr = Path(TEMP_CF_OUT) / "final_results"
    direct = fr / f"{stem}_fake_B.png"
    if direct.exists(): return str(direct)
    candidates = sorted(fr.glob("*.png"), key=os.path.getmtime, reverse=True) if fr.exists() else []
    return str(candidates[0]) if candidates else None

def run_pipeline(sketch_bgr, stem, fidelity):
    if os.path.exists(TEMP_VAL_DIR): shutil.rmtree(TEMP_VAL_DIR)
    save_pix2pix_input(sketch_bgr, f"{stem}.png")

    with st.spinner("Running Pix2Pix…"):
        r = run_cmd([
            PYTHON_EXE, "test.py",
            "--dataroot", TEMP_TEST_ROOT, "--name", EXP_NAME,
            "--model", "pix2pix", "--direction", "AtoB",
            "--input_nc", "3", "--output_nc", "3",
            "--checkpoints_dir", CHECKPOINTS, "--results_dir", PIX2PIX_RESULTS,
            "--phase", "val", "--load_size", "256", "--crop_size", "256",
            "--preprocess", "none", "--no_flip", "--netG", "unet_256",
            "--norm", "batch", "--epoch", "latest", "--eval",
            "--num_test", "100", "--num_threads", "0",
            "--batch_size", "1", "--serial_batches",
        ], cwd=PIX2PIX_REPO)

    if r.returncode != 0:
        st.error("Pix2Pix failed.")
        with st.expander("Show error log"): st.code(r.stderr[-2000:])
        st.stop()

    pix_path = find_pix2pix_output(stem)
    if not pix_path:
        st.error("Output image not found after model ran.")
        st.stop()

    pix_img = np.array(Image.open(pix_path).convert("RGB"))
    cf_img  = None

    with st.spinner(f"CodeFormer enhancing (fidelity {fidelity:.2f})…"):
        cf_input_dir = os.path.join(BASE_TEMP, "cf_input")
        if os.path.exists(cf_input_dir): shutil.rmtree(cf_input_dir)
        os.makedirs(cf_input_dir)
        fake_b_src = Path(PIX2PIX_RESULTS) / EXP_NAME / "val_latest" / "images" / f"{stem}_fake_B.png"
        if fake_b_src.exists():
            shutil.copy(str(fake_b_src), os.path.join(cf_input_dir, f"{stem}_fake_B.png"))
        cf = run_cmd([
            PYTHON_EXE, "inference_codeformer.py",
            "--input_path", cf_input_dir, "--output_path", TEMP_CF_OUT,
            "--fidelity_weight", str(fidelity),
            *( ["--face_upsample"] if FACE_UPSAMPLE else [] ),
        ], cwd=CODEFORMER_REPO)

    if cf.returncode == 0:
        cf_path = find_cf_output(stem)
        if cf_path:
            cf_img = np.array(Image.open(cf_path).convert("RGB").resize((256, 256)))

    return pix_img, cf_img

# ── Visual helpers ────────────────────────────────────────────────

def sketch_heatmap(sketch_bgr):
    gray    = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2GRAY)
    inv     = cv2.bitwise_not(gray)
    density = cv2.GaussianBlur(inv.astype(np.float32), (21, 21), 0)
    density = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(density, cv2.COLORMAP_JET)
    sketch3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(sketch3, 0.35, heatmap, 0.65, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

def diff_heatmap(pix_img_rgb, cf_img_rgb):
    diff    = np.abs(pix_img_rgb.astype(np.float32) - cf_img_rgb.astype(np.float32)).mean(axis=2)
    diff_u8 = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    diff_u8 = cv2.GaussianBlur(diff_u8, (5, 5), 0)
    heatmap = cv2.applyColorMap(diff_u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

def show_before_after(before_rgb, after_rgb, label_left="Sketch", label_right="Generated"):
    b64_before = img_to_b64(before_rgb)
    b64_after  = img_to_b64(after_rgb)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: transparent; font-family: Inter, sans-serif; overflow: hidden; }}
  .wrap {{
    position: relative;
    width: 100%;
    border-radius: 12px;
    overflow: hidden;
    cursor: col-resize;
    user-select: none;
    border: 1px solid #ece5dc;
    background: #f5f1eb;
    line-height: 0;
  }}
  .back-img {{
    display: block;
    width: 100%;
    height: auto;
  }}
  .clip {{
    position: absolute;
    top: 0; left: 0;
    height: 100%;
    overflow: hidden;
    width: 50%;
  }}
  .front-img {{
    display: block;
    position: absolute;
    top: 0; left: 0;
    height: 100%;
    max-width: none;
  }}
  .divider {{
    position: absolute;
    top: 0; left: 50%;
    width: 3px;
    height: 100%;
    background: #d6b892;
    transform: translateX(-50%);
    z-index: 10;
    pointer-events: none;
  }}
  .handle {{
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 40px; height: 40px;
    background: #d6b892;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 15px; font-weight: 700;
    box-shadow: 0 2px 10px rgba(0,0,0,0.22);
    pointer-events: none;
  }}
  .lbl {{
    position: absolute; bottom: 10px;
    font-size: 0.62rem; letter-spacing: 0.07em; text-transform: uppercase;
    background: rgba(255,255,255,0.88); padding: 3px 8px;
    border-radius: 6px; color: #6b6259; pointer-events: none;
    line-height: 1.4;
  }}
  .lbl-l {{ left: 10px; }}
  .lbl-r {{ right: 10px; }}
</style>
</head>
<body>
<div class="wrap" id="wrap">
  <!-- base layer: generated face (full width) -->
  <img class="back-img" id="back" src="data:image/png;base64,{b64_after}" />
  <!-- clip layer: sketch revealed on right side -->
  <div class="clip" id="clip">
    <img class="front-img" id="front" src="data:image/png;base64,{b64_before}" />
  </div>
  <div class="divider" id="div">
    <div class="handle">&#x21D4;</div>
  </div>
  <span class="lbl lbl-l">{label_right}</span>
  <span class="lbl lbl-r">{label_left}</span>
</div>

<script>
  var wrap  = document.getElementById('wrap');
  var clip  = document.getElementById('clip');
  var front = document.getElementById('front');
  var div   = document.getElementById('div');
  var drag  = false;

  function reportHeight() {{
    var h = wrap.offsetHeight;
    if (h > 10) {{
      window.parent.postMessage({{ type: 'streamlit:setFrameHeight', height: h + 8 }}, '*');
    }}
  }}

  function syncFrontWidth() {{
    front.style.width  = wrap.offsetWidth + 'px';
    front.style.height = wrap.offsetHeight + 'px';
    front.style.objectFit = 'cover';
  }}

  function setPos(clientX) {{
    var rect = wrap.getBoundingClientRect();
    var pct  = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
    clip.style.width = (pct * 100) + '%';
    div.style.left   = (pct * 100) + '%';
  }}

  // ResizeObserver fires after layout paint — much more reliable than load event
  var ro = new ResizeObserver(function(entries) {{
    syncFrontWidth();
    reportHeight();
  }});
  ro.observe(wrap);

  document.getElementById('back').addEventListener('load', function() {{
    syncFrontWidth();
    setPos(wrap.getBoundingClientRect().left + wrap.offsetWidth * 0.5);
    reportHeight();
    // Re-report after a short delay to catch any late reflows
    setTimeout(reportHeight, 200);
    setTimeout(reportHeight, 600);
  }});

  wrap.addEventListener('mousedown',  function(e) {{ drag = true; setPos(e.clientX); e.preventDefault(); }});
  wrap.addEventListener('touchstart', function(e) {{ drag = true; setPos(e.touches[0].clientX); }}, {{passive: true}});
  window.addEventListener('mousemove',  function(e) {{ if (drag) setPos(e.clientX); }});
  window.addEventListener('touchmove',  function(e) {{ if (drag) setPos(e.touches[0].clientX); }}, {{passive: true}});
  window.addEventListener('mouseup',  function() {{ drag = false; }});
  window.addEventListener('touchend', function() {{ drag = false; }});
  window.addEventListener('resize',   function() {{ syncFrontWidth(); reportHeight(); }});
</script>
</body>
</html>
"""
    components.html(html, height=800, scrolling=False)


def show_pipeline_cards(sketch_rgb, pix_img, cf_img):
    best   = cf_img if cf_img is not None else pix_img
    imgs   = [
        (sketch_rgb, "Sketch",   False),
        (pix_img,    "Pix2Pix",  False),
        (best,       "Enhanced", True),
    ]
    cols = st.columns(3)
    for col, (img, label, glow) in zip(cols, imgs):
        b64 = img_to_b64(img)
        glow_style = (
            "animation: glow-pulse 2.4s ease-in-out infinite; border: 1.5px solid #d6b892;"
            if glow else "border: 1px solid #ece5dc;"
        )
        html = f"""
<!DOCTYPE html><html><head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: transparent; font-family: Inter, sans-serif; overflow: hidden; }}
  @keyframes fade-up {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}
  @keyframes glow-pulse {{
    0%   {{ box-shadow: 0 0 10px 2px rgba(201,170,131,0.4); }}
    50%  {{ box-shadow: 0 0 26px 6px rgba(201,170,131,0.7); }}
    100% {{ box-shadow: 0 0 10px 2px rgba(201,170,131,0.4); }}
  }}
  .card {{
    border-radius: 12px; overflow: hidden;
    animation: fade-up 0.6s cubic-bezier(0.22,1,0.36,1) both;
    {glow_style}
  }}
  img {{ display: block; width: 100%; height: auto; border-radius: 0; }}
  .lbl {{
    font-size: 0.68rem; color: #9a8f82; text-align: center;
    margin-top: 6px; letter-spacing: 0.06em; text-transform: uppercase;
  }}
</style>
</head><body>
  <div class="card" id="card"><img id="img" src="data:image/png;base64,{b64}" /></div>
  <p class="lbl">{label}</p>
<script>
  document.getElementById('img').addEventListener('load', function() {{
    var h = document.body.scrollHeight;
    window.parent.postMessage({{ type: 'streamlit:setFrameHeight', height: h + 4 }}, '*');
  }});
</script>
</body></html>
"""
        with col:
            components.html(html, height=320, scrolling=False)


def show_results_enhanced(sketch_rgb, pix_img, cf_img):
    st.divider()

    # 1. Progressive pipeline with fade + glow
    st.markdown('<p class="section-header">Pipeline · Step by Step</p>', unsafe_allow_html=True)
    show_pipeline_cards(sketch_rgb, pix_img, cf_img)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Before / After wipe slider
    st.markdown('<p class="section-header">Before · After &nbsp;<span style="font-weight:400;font-size:0.68rem;color:#b8aa9a">drag to reveal</span></p>', unsafe_allow_html=True)
    best = cf_img if cf_img is not None else pix_img
    show_before_after(sketch_rgb, best, label_left="Sketch", label_right="Generated")

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Diff heatmap
    if cf_img is not None:
        st.markdown('<p class="section-header">Enhancement · Difference Map</p>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            st.image(pix_img, use_container_width=True)
            st.markdown('<p class="img-label">Pix2Pix</p>', unsafe_allow_html=True)
        with d2:
            st.image(diff_heatmap(pix_img, cf_img), use_container_width=True)
            st.markdown('<p class="diff-caption">Bright = most changed by CodeFormer</p>', unsafe_allow_html=True)
        with d3:
            st.image(cf_img, use_container_width=True)
            st.markdown('<p class="img-label">Enhanced</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Downloads
    dl_imgs = [(sketch_rgb, "Sketch"), (pix_img, "Pix2Pix")]
    if cf_img is not None: dl_imgs.append((cf_img, "Enhanced"))
    dcols = st.columns(len(dl_imgs))
    for col, (img, label) in zip(dcols, dl_imgs):
        with col:
            col.download_button(f"↓  {label}", to_png_bytes(img),
                                f"result_{label.lower()}.png", "image/png",
                                use_container_width=True)


def show_consistency(original_sketch_bgr, generated_face_rgb):
    with st.spinner("Running consistency check…"):
        face_bgr   = cv2.cvtColor(generated_face_rgb, cv2.COLOR_RGB2BGR)
        face_bgr   = cv2.resize(face_bgr, (256, 256))
        recon_bgr  = make_sketch(face_bgr)
        orig_gray  = cv2.cvtColor(original_sketch_bgr, cv2.COLOR_BGR2GRAY)
        recon_gray = cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2GRAY)
        score      = round(float(ssim(orig_gray, recon_gray, data_range=255)) * 100, 1)
        recon_rgb  = cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2RGB)

    if score >= 70:
        verdict, color = "High consistency — reconstruction reliably matches the sketch.", "#4a7c59"
    elif score >= 50:
        verdict, color = "Moderate consistency — minor structural differences detected.", "#b38a5d"
    else:
        verdict, color = "Low consistency — generated face may diverge from the sketch.", "#a0503a"

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Sketch · Consistency Check</p>', unsafe_allow_html=True)
    c_score, c_img = st.columns([1, 2])
    with c_score:
        st.markdown(f"""
        <div class="consistency-card">
            <div class="consistency-title">Reconstruction Score</div>
            <div class="consistency-score" style="color:{color}">{score}%</div>
            <div class="consistency-verdict">{verdict}</div>
        </div>""", unsafe_allow_html=True)
    with c_img:
        st.image(recon_rgb, use_container_width=True)
        st.markdown('<p class="img-label">Re-sketched Face</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# Page
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; padding-bottom: 0.8rem; border-bottom: 1px solid #e6ddd2;">
    <a href="http://localhost:8000" target="_self" style="text-decoration: none; color: #b38a5d; font-weight: 500; font-size: 0.85rem; font-family: 'Inter', sans-serif; display: flex; align-items: center; gap: 4px; transition: color 0.2s;">
        ← Back to Landing Page
    </a>
    <span style="font-size: 0.72rem; color: #9a8f82; font-family: 'Inter', sans-serif; letter-spacing: 0.05em; text-transform: uppercase; font-weight: 500;">
        AI335L Lab Capstone
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown('<p class="site-title">Sketch to Face</p>', unsafe_allow_html=True)
st.markdown('<p class="site-subtitle">Generate a photorealistic face from a portrait photo or a hand-drawn sketch.</p>', unsafe_allow_html=True)

tab_photo, tab_sketch = st.tabs(["From Photo", "From Sketch"])

with tab_photo:
    uploaded_photo = st.file_uploader("photo", type=["jpg","jpeg","png","webp"],
                                      label_visibility="collapsed", key="photo_upload")
    if uploaded_photo:
        raw      = np.frombuffer(uploaded_photo.read(), np.uint8)
        bgr_orig = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        cropped = crop_face(bgr_orig)
        if cropped is None:
            st.error("No face detected in the uploaded photo. Please upload a clear, front-facing portrait and try again.")
            st.stop()
        bgr_256 = cv2.resize(cropped, (256, 256))

        with st.spinner("Generating sketch…"):
            sketch_bgr = make_sketch(bgr_256)
        sketch_rgb = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2RGB)

        c1, gap, c2 = st.columns([10, 1, 10])
        with c1:
            st.image(cv2.cvtColor(bgr_256, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('<p class="img-label">Photo</p>', unsafe_allow_html=True)
        with gap:
            st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)
        with c2:
            st.image(sketch_rgb, use_container_width=True)
            st.markdown('<p class="img-label">Sketch</p>', unsafe_allow_html=True)

        with st.expander("🔍 Sketch confidence heatmap"):
            st.image(sketch_heatmap(sketch_bgr), use_container_width=True)
            st.caption("Warm = dense linework · Cool = sparse / uncertain regions")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="fidelity-label">CodeFormer fidelity — lower = more restoration · higher = stays closer to Pix2Pix output</p>', unsafe_allow_html=True)
        fidelity = st.slider("fidelity", 0.0, 1.0, 0.7, 0.05, label_visibility="collapsed", key="fidelity_photo")

        if st.button("Generate Face", type="primary", use_container_width=True, key="btn_photo"):
            stem = Path(uploaded_photo.name).stem
            pix_img, cf_img = run_pipeline(sketch_bgr, stem, fidelity)
            show_results_enhanced(sketch_rgb, pix_img, cf_img)
            best = cf_img if cf_img is not None else pix_img
            show_consistency(sketch_bgr, best)

with tab_sketch:
    st.caption("Upload a lineart sketch — black lines on white background, square crop preferred.")
    uploaded_sketch = st.file_uploader("sketch", type=["jpg","jpeg","png","webp"],
                                       label_visibility="collapsed", key="sketch_upload")
    if uploaded_sketch:
        raw        = np.frombuffer(uploaded_sketch.read(), np.uint8)
        sketch_raw = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        sketch_bgr = cv2.resize(sketch_raw, (256, 256))
        sketch_rgb = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2RGB)

        st.image(sketch_rgb, use_container_width=False, width=256)
        st.markdown('<p class="img-label">Sketch</p>', unsafe_allow_html=True)

        with st.expander("🔍 Sketch confidence heatmap"):
            st.image(sketch_heatmap(sketch_bgr), use_container_width=True)
            st.caption("Warm = dense linework · Cool = sparse / uncertain regions")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="fidelity-label">CodeFormer fidelity — lower = more restoration · higher = stays closer to Pix2Pix output</p>', unsafe_allow_html=True)
        fidelity = st.slider("fidelity", 0.0, 1.0, 0.7, 0.05, label_visibility="collapsed", key="fidelity_sketch")

        if st.button("Generate Face", type="primary", use_container_width=True, key="btn_sketch"):
            stem = Path(uploaded_sketch.name).stem
            pix_img, cf_img = run_pipeline(sketch_bgr, stem, fidelity)
            show_results_enhanced(sketch_rgb, pix_img, cf_img)
            best = cf_img if cf_img is not None else pix_img
            show_consistency(sketch_bgr, best)