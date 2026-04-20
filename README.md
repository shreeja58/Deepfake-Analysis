# 🛡️ DeepShield AI
### Robust Deepfake Analysis & Authentication Platform for Social Media Protection

---

## 📁 Project Structure

```
deepshield/
├── app.py                  ← Main Streamlit web application
├── visual_analysis.py      ← Blur, edge, noise & face detection module
├── biological_analysis.py  ← Skin tone, eye region & landmark module
├── audio_meta_analysis.py  ← EXIF, DCT frequency & SHA-256 module
├── scoring_engine.py       ← Ensemble confidence & verdict engine
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.9 or above
- pip (Python package manager)

### 2. Install Dependencies
Open terminal in the `deepshield/` folder and run:

```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

The browser will automatically open at: **http://localhost:8501**

---

## 🚀 How to Use

1. Open the web app in your browser
2. Upload any image (JPG, PNG, WEBP, BMP — max 10 MB)
3. The face detection overlay will appear automatically
4. Click **"RUN DEEPFAKE ANALYSIS"**
5. View the verdict, confidence score, and detailed module breakdown

---

## 🔬 How It Works

### Module 1 — Visual Analysis (Weight: 40%)
| Technique | What it checks |
|-----------|----------------|
| Laplacian Blur | Sharpness variance — deepfakes blur face boundaries |
| Canny Edge | Edge density — composited images show abnormal edges |
| Noise Analysis | Pixel noise patterns — AI images lack natural camera noise |
| Face Symmetry | Facial symmetry ratio via OpenCV Haar cascades |

### Module 2 — Biological Signals (Weight: 35%)
| Technique | What it checks |
|-----------|----------------|
| Skin Tone Uniformity | YCrCb skin pixel variance — GAN faces are unnaturally smooth |
| Eye Region | Eye detection & alignment — deepfakes misplace eye geometry |
| Landmark Consistency | Sobel gradient structure — GAN faces have diffuse gradients |

### Module 3 — Audio / Metadata (Weight: 25%)
| Technique | What it checks |
|-----------|----------------|
| EXIF Metadata | Camera make/model/datetime — AI images lack rich EXIF |
| DCT Frequency | Block frequency energy — GAN fingerprints in frequency domain |
| Compression | JPEG block boundary consistency — double-compression = splicing |
| SHA-256 Hash | Unique file fingerprint for integrity verification |

### Scoring Formula
```
Confidence = 0.40 × Visual + 0.35 × Biological + 0.25 × Audio/Meta
```

### Verdict Classification
```
≥ 70%  →  ✅ AUTHENTIC   — Safe to publish
45–69% →  ⚠️ SUSPICIOUS  — Request source verification
< 45%  →  🚨 DEEPFAKE    — Do NOT publish
```

---

## 🛠️ Technologies Used

| Technology | Role |
|------------|------|
| Python 3.x | Core language |
| Streamlit | Web UI framework |
| OpenCV | Computer vision & image processing |
| NumPy | Numerical computation |
| Pillow (PIL) | EXIF metadata extraction |
| SHA-256 (hashlib) | File integrity hashing |

---

## 🔮 Future Enhancements

- Integrate CNN / Vision Transformer models for higher accuracy
- Full video-frame deepfake detection (frame-by-frame analysis)
- Real-time audio AI analysis using `librosa`
- Browser extension for instant social media verification
- REST API for platform integration

---

## ⚠️ Disclaimer

DeepShield AI is an academic prototype. Results are probabilistic and should be
combined with human expert review for high-stakes decisions. Not intended for
production deployment without further validation.

---

*Mini Project · Department of Computer Science · 2024–25*
