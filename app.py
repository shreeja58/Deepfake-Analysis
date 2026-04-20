"""
DeepShield AI — Main Streamlit Application
Robust Deepfake Analysis & Authentication Platform for Social Media Protection

Run: streamlit run app.py
"""

import io
import time
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# DeepShield modules
from visual_analysis    import run_visual_analysis
from biological_analysis import run_biological_analysis
from audio_meta_analysis import run_audio_meta_analysis
from scoring_engine      import compute_confidence

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A1628;
    color: #E3F2FD;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0D2140;
    border-right: 1px solid #1A3A6A;
}

/* ── Header ── */
.ds-header {
    background: linear-gradient(135deg, #0D2140 0%, #0A1628 100%);
    border: 1px solid #00B0FF22;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.ds-header h1 {
    font-family: 'Orbitron', monospace;
    font-size: 2.6rem;
    font-weight: 900;
    color: #00E5FF;
    margin: 0;
    letter-spacing: 2px;
    text-shadow: 0 0 30px #00E5FF55;
}
.ds-header p {
    color: #90CAF9;
    margin: 0.3rem 0 0 0;
    font-size: 1rem;
    font-weight: 300;
}

/* ── Score Card ── */
.score-card {
    background: #0D2140;
    border-radius: 14px;
    border: 1px solid #1A3A6A;
    padding: 1.4rem;
    text-align: center;
    margin-bottom: 1rem;
    transition: transform 0.2s;
}
.score-card:hover { transform: translateY(-2px); }
.score-number {
    font-family: 'Orbitron', monospace;
    font-size: 2.6rem;
    font-weight: 900;
    line-height: 1;
}
.score-label {
    font-size: 0.78rem;
    color: #64B5F6;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.4rem;
}

/* ── Verdict Banner ── */
.verdict-authentic { background: linear-gradient(135deg,#003300,#004d00); border:2px solid #00C853; border-radius:16px; padding:1.5rem 2rem; }
.verdict-suspicious { background: linear-gradient(135deg,#1a0a00,#2a1800); border:2px solid #FF9100; border-radius:16px; padding:1.5rem 2rem; }
.verdict-deepfake   { background: linear-gradient(135deg,#1a0000,#2a0000); border:2px solid #FF1744; border-radius:16px; padding:1.5rem 2rem; }

.verdict-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.9rem;
    font-weight: 900;
    letter-spacing: 3px;
}
.verdict-rec {
    margin-top: 0.5rem;
    font-size: 0.95rem;
    color: #ccc;
}

/* ── Detail Row ── */
.detail-row {
    background: #0D2140;
    border-left: 3px solid #00B0FF;
    border-radius: 0 10px 10px 0;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    color: #B0BEC5;
}
.detail-row strong { color: #E3F2FD; }

/* ── Progress Bar ── */
.bar-wrap {
    background: #0D2140;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    border: 1px solid #1A3A6A;
}
.bar-label { font-size: 0.82rem; color: #90CAF9; margin-bottom: 0.4rem; display:flex; justify-content:space-between; }
.bar-bg { background: #1A3A6A; border-radius: 6px; height: 10px; overflow:hidden; }
.bar-fill { height: 10px; border-radius: 6px; transition: width 0.6s ease; }

/* ── Module Section ── */
.module-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.9rem;
    letter-spacing: 2px;
    color: #00E5FF;
    text-transform: uppercase;
    padding: 0.5rem 0;
    border-bottom: 1px solid #00B0FF33;
    margin-bottom: 1rem;
}

/* ── Hash Box ── */
.hash-box {
    background: #050E1A;
    border: 1px solid #00B0FF33;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: 'Courier New', monospace;
    font-size: 0.72rem;
    color: #00E5FF;
    word-break: break-all;
    margin-top: 0.5rem;
}

/* ── Upload Zone ── */
.upload-info {
    background: #0D2140;
    border: 1px dashed #00B0FF55;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    color: #64B5F6;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

/* ── Streamlit overrides ── */
.stButton button {
    background: linear-gradient(135deg, #0047AB, #00B0FF) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.9rem !important;
    letter-spacing: 1.5px !important;
    padding: 0.6rem 1.8rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.85 !important; }
div[data-testid="stFileUploader"] { border: none !important; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def score_color(score: float) -> str:
    if score >= 70: return "#00C853"
    if score >= 45: return "#FF9100"
    return "#FF1744"


def progress_bar_html(label: str, score: float, detail: str = "") -> str:
    color = score_color(score)
    return f"""
    <div class="bar-wrap">
        <div class="bar-label">
            <span>{label}</span>
            <span style="color:{color};font-weight:600">{score:.1f}%</span>
        </div>
        {'<div style="font-size:0.76rem;color:#78909C;margin-bottom:0.35rem">' + detail + '</div>' if detail else ''}
        <div class="bar-bg">
            <div class="bar-fill" style="width:{score}%;background:{color}"></div>
        </div>
    </div>
    """


def verdict_html(verdict: str, confidence: float, emoji: str, risk: str, rec: str, color: str) -> str:
    css_class = f"verdict-{verdict.lower()}"
    return f"""
    <div class="{css_class}">
        <div class="verdict-title" style="color:{color}">{emoji} {verdict}</div>
        <div style="font-family:'Orbitron',monospace;font-size:1rem;color:{color};margin-top:0.3rem">
            Confidence: {confidence:.1f}% &nbsp;|&nbsp; {risk}
        </div>
        <div class="verdict-rec">{rec}</div>
    </div>
    """


def draw_face_boxes(image_bytes: bytes) -> Image.Image:
    """Draw face bounding boxes on a copy of the image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 229, 255), 2)
        cv2.putText(img, "FACE", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 255), 2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ DeepShield AI")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "DeepShield AI is a deepfake detection platform that analyzes images "
        "using visual, biological, and metadata signals to produce an ensemble authenticity score."
    )
    st.markdown("---")
    st.markdown("**Scoring Formula**")
    st.code("Confidence =\n  0.40 × Visual\n+ 0.35 × Biological\n+ 0.25 × Audio/Meta", language="text")
    st.markdown("---")
    st.markdown("**Verdict Thresholds**")
    st.markdown("🟢 **≥ 70%** → Authentic")
    st.markdown("🟡 **45–69%** → Suspicious")
    st.markdown("🔴 **< 45%** → Deepfake")
    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("`Python` · `Streamlit` · `OpenCV` · `NumPy` · `Pillow` · `SHA-256`")
    st.markdown("---")
    st.caption("Mini Project · Dept. of Computer Science · 2024–25")


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ds-header">
    <div style="font-size:3.5rem">🛡️</div>
    <div>
        <h1>DeepShield AI</h1>
        <p>Robust Deepfake Analysis &amp; Authentication Platform for Social Media Protection</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Upload ───────────────────────────────────────────────────────────────────
col_upload, col_info = st.columns([1.2, 1])

with col_upload:
    st.markdown('<div class="module-header">📤 Upload Media</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-info">Supported: JPG · PNG · WEBP · BMP<br>Max size: 10 MB</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png", "webp", "bmp"], label_visibility="collapsed")

with col_info:
    st.markdown('<div class="module-header">ℹ️ How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="detail-row"><strong>Step 1 →</strong> Upload a media file</div>
    <div class="detail-row"><strong>Step 2 →</strong> Visual analysis (blur, edges, noise, faces)</div>
    <div class="detail-row"><strong>Step 3 →</strong> Biological signals (skin, eyes, landmarks)</div>
    <div class="detail-row"><strong>Step 4 →</strong> Metadata & integrity (EXIF, DCT, SHA-256)</div>
    <div class="detail-row"><strong>Step 5 →</strong> Ensemble confidence score + verdict</div>
    """, unsafe_allow_html=True)


# ─── Analysis ─────────────────────────────────────────────────────────────────
if uploaded is not None:
    image_bytes = uploaded.read()

    col_img, col_face = st.columns(2)
    with col_img:
        st.markdown('<div class="module-header">🖼️ Original Image</div>', unsafe_allow_html=True)
        st.image(image_bytes, use_container_width=True)

    with col_face:
        st.markdown('<div class="module-header">🔍 Face Detection Overlay</div>', unsafe_allow_html=True)
        annotated = draw_face_boxes(image_bytes)
        st.image(annotated, use_container_width=True)

    st.markdown("---")

    if st.button("🔬  RUN DEEPFAKE ANALYSIS"):
        progress = st.progress(0, text="Initializing DeepShield AI...")
        time.sleep(0.3)

        # ── Visual Analysis ──
        progress.progress(15, text="Running visual analysis (blur · edges · noise · faces)...")
        time.sleep(0.4)
        try:
            vis_result = run_visual_analysis(image_bytes)
        except Exception as e:
            st.error(f"Visual analysis error: {e}")
            st.stop()

        progress.progress(40, text="Running biological signal analysis...")
        time.sleep(0.4)
        try:
            bio_result = run_biological_analysis(image_bytes)
        except Exception as e:
            st.error(f"Biological analysis error: {e}")
            st.stop()

        progress.progress(65, text="Analyzing metadata · DCT frequency · SHA-256...")
        time.sleep(0.4)
        try:
            meta_result = run_audio_meta_analysis(image_bytes)
        except Exception as e:
            st.error(f"Metadata analysis error: {e}")
            st.stop()

        progress.progress(85, text="Computing ensemble confidence score...")
        time.sleep(0.3)

        verdict_obj = compute_confidence(
            visual_score=vis_result.overall_visual_score,
            biological_score=bio_result.overall_biological_score,
            audio_score=meta_result.overall_audio_score,
        )

        progress.progress(100, text="Analysis complete!")
        time.sleep(0.3)
        progress.empty()

        st.success("✅ Analysis complete!")

        # ── Verdict Banner ──
        st.markdown("## 🎯 Verdict")
        st.markdown(verdict_html(
            verdict_obj.verdict,
            verdict_obj.confidence,
            verdict_obj.verdict_emoji,
            verdict_obj.risk_level,
            verdict_obj.recommendation,
            verdict_obj.verdict_color,
        ), unsafe_allow_html=True)

        st.markdown("---")

        # ── Score Columns ──
        st.markdown("## 📊 Module Scores")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-number" style="color:{score_color(verdict_obj.confidence)}">{verdict_obj.confidence:.1f}%</div>
                <div class="score-label">Final Confidence</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-number" style="color:{score_color(vis_result.overall_visual_score)}">{vis_result.overall_visual_score:.1f}%</div>
                <div class="score-label">Visual (40%)</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-number" style="color:{score_color(bio_result.overall_biological_score)}">{bio_result.overall_biological_score:.1f}%</div>
                <div class="score-label">Biological (35%)</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-number" style="color:{score_color(meta_result.overall_audio_score)}">{meta_result.overall_audio_score:.1f}%</div>
                <div class="score-label">Audio/Meta (25%)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Module Details ──
        col_v, col_b, col_a = st.columns(3)

        with col_v:
            st.markdown('<div class="module-header">👁️ Visual Analysis</div>', unsafe_allow_html=True)
            vd = vis_result.details
            st.markdown(progress_bar_html("Blur (Laplacian)", vd["blur"]["score"], vd["blur"]["label"]), unsafe_allow_html=True)
            st.markdown(progress_bar_html("Edge Consistency (Canny)", vd["edges"]["score"], vd["edges"]["label"]), unsafe_allow_html=True)
            st.markdown(progress_bar_html("Noise Pattern", vd["noise"]["score"], vd["noise"]["label"]), unsafe_allow_html=True)
            st.markdown(progress_bar_html("Face Symmetry", vd["face"]["symmetry_score"], vd["face"]["label"]), unsafe_allow_html=True)
            faces_found = f"{'✅' if vd['face']['detected'] else '❌'} {vd['face']['count']} face(s) detected"
            st.markdown(f'<div class="detail-row">{faces_found}</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="module-header">🧬 Biological Signals</div>', unsafe_allow_html=True)
            bd = bio_result.details
            st.markdown(progress_bar_html("Skin Tone Uniformity", bd["skin_tone"]["score"], bd["skin_tone"]["label"]), unsafe_allow_html=True)
            st.markdown(progress_bar_html("Eye Region Analysis", bd["eye_region"]["score"], bd["eye_region"]["label"]), unsafe_allow_html=True)
            st.markdown(progress_bar_html("Landmark Consistency", bd["landmarks"]["score"], bd["landmarks"]["label"]), unsafe_allow_html=True)
            face_found_str = "✅ Face region found" if bd["face_found"] else "❌ No face region"
            st.markdown(f'<div class="detail-row">{face_found_str}</div>', unsafe_allow_html=True)

        with col_a:
            st.markdown('<div class="module-header">🔐 Integrity & Metadata</div>', unsafe_allow_html=True)
            ad = meta_result.details
            st.markdown(progress_bar_html("EXIF Metadata", meta_result.exif_score, ad["exif"].get("label", "")), unsafe_allow_html=True)
            st.markdown(progress_bar_html("DCT Frequency", meta_result.dct_score, ad["dct"]["label"]), unsafe_allow_html=True)
            st.markdown(progress_bar_html("Compression Integrity", meta_result.compression_score, ad["compression"]["label"]), unsafe_allow_html=True)
            st.markdown(f'<div class="module-header" style="font-size:0.75rem;margin-top:1rem">SHA-256 Hash</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="hash-box">{meta_result.hash_sha256}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Breakdown Table ──
        st.markdown("## 🧮 Scoring Breakdown")
        bk = verdict_obj.breakdown
        contrib = bk["contributions"]
        labels  = bk["module_labels"]
        breakdown_data = {
            "Module":       ["🎯 Visual",         "🧬 Biological",         "🔐 Audio/Meta"],
            "Weight":       ["40%",               "35%",                   "25%"],
            "Raw Score":    [f"{vis_result.overall_visual_score:.1f}%",
                             f"{bio_result.overall_biological_score:.1f}%",
                             f"{meta_result.overall_audio_score:.1f}%"],
            "Contribution": [f"{contrib['visual']:.2f}",
                             f"{contrib['biological']:.2f}",
                             f"{contrib['audio']:.2f}"],
            "Assessment":   [labels["visual"], labels["biological"], labels["audio"]],
        }
        st.table(breakdown_data)
        st.markdown(f"""
        <div style="text-align:center;font-family:'Orbitron',monospace;color:#00E5FF;font-size:1.1rem;margin-top:0.5rem">
            Final Confidence = {contrib['visual']:.2f} + {contrib['biological']:.2f} + {contrib['audio']:.2f} = <strong>{verdict_obj.confidence:.2f}%</strong>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption(
            "⚠️ DeepShield AI is a prototype system for academic use. "
            "Results are probabilistic and should be combined with human review for high-stakes decisions."
        )
else:
    st.markdown("""
    <div style="text-align:center;padding:3rem;color:#546E7A;font-size:1.1rem;">
        🛡️ Upload an image above to begin deepfake analysis
    </div>
    """, unsafe_allow_html=True)
