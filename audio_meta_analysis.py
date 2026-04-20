"""
DeepShield AI — Audio / Metadata Analysis Module
For image-only inputs, this module analyzes EXIF metadata, compression
artifacts, and DCT frequency distribution as a proxy "audio-equivalent"
integrity signal (weighted 25% in the ensemble).

For real video/audio files, this module can be extended with librosa-based
spectral analysis (commented stubs provided at bottom).
"""

import io
import hashlib
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class AudioMetaResult:
    hash_sha256: str
    exif_score: float          # 0-100
    dct_score: float           # 0-100 (frequency domain integrity)
    compression_score: float   # 0-100
    overall_audio_score: float # 0-100 (final)
    details: dict


def compute_sha256(image_bytes: bytes) -> str:
    """SHA-256 hash for file integrity verification."""
    return hashlib.sha256(image_bytes).hexdigest()


def analyze_exif_metadata(image_bytes: bytes) -> Tuple[float, dict]:
    """
    EXIF metadata analysis.
    Authentic camera images typically carry camera model, timestamp, GPS, etc.
    AI-generated / deepfake images often lack EXIF or have stripped metadata.
    """
    if not PIL_AVAILABLE:
        return 50.0, {"note": "Pillow not installed — EXIF analysis skipped"}

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        exif_data = pil_img._getexif() if hasattr(pil_img, "_getexif") else None
    except Exception:
        return 40.0, {"error": "Could not read EXIF"}

    if exif_data is None:
        return 35.0, {"exif": "No EXIF data — possible AI generation or stripped metadata"}

    tags = {}
    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)
        tags[str(tag)] = str(value)[:80]  # truncate long values

    has_camera   = "Make" in tags or "Model" in tags
    has_datetime = "DateTime" in tags or "DateTimeOriginal" in tags
    has_software = "Software" in tags
    tag_count    = len(tags)

    # Score based on EXIF richness
    if has_camera and has_datetime and tag_count >= 10:
        score = 90.0
        label = "Rich EXIF — likely authentic camera image"
    elif has_camera or has_datetime:
        score = 68.0
        label = "Partial EXIF — some metadata present"
    elif has_software:
        score = 30.0
        label = "Software EXIF only — edited or generated image"
    else:
        score = 40.0
        label = "Minimal EXIF — metadata possibly stripped"

    return round(score, 2), {"label": label, "tag_count": tag_count, "sample_tags": dict(list(tags.items())[:6])}


def analyze_dct_frequency(image_bytes: bytes) -> Tuple[float, str]:
    """
    DCT (Discrete Cosine Transform) frequency domain analysis.
    GAN-generated images often show distinct high-frequency artifacts
    ('GAN fingerprints') not present in authentic photographs.
    We analyze energy distribution across frequency bands.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 50.0, "Could not decode image for DCT"

    # Resize to 256x256 for consistent analysis
    img_resized = cv2.resize(img, (256, 256)).astype(np.float32)

    # Block-wise 8x8 DCT (same as JPEG)
    dct_coeffs = []
    for i in range(0, 256, 8):
        for j in range(0, 256, 8):
            block = img_resized[i:i+8, j:j+8]
            dct_block = cv2.dct(block)
            dct_coeffs.append(dct_block.flatten())

    dct_array = np.array(dct_coeffs)

    # High-frequency energy ratio (coefficients 32-63 in 8x8 block = indices 32:)
    low_energy  = np.abs(dct_array[:, :32]).mean()
    high_energy = np.abs(dct_array[:, 32:]).mean()
    hf_ratio = high_energy / (low_energy + 1e-6)

    # Authentic photos: balanced HF/LF, ratio typically 0.05–0.20
    # GAN images: unusually low HF (over-smooth) or periodic spikes
    std_across_blocks = np.std(dct_array[:, 1:4])  # DC neighborhood variance

    if 0.04 <= hf_ratio <= 0.22 and std_across_blocks > 5:
        score = 85.0
        label = "Natural DCT frequency distribution"
    elif hf_ratio < 0.03:
        score = 25.0
        label = "Abnormally low HF energy — GAN over-smoothing"
    elif hf_ratio > 0.30:
        score = 40.0
        label = "Excessive HF energy — heavy compression or splicing"
    else:
        score = 60.0
        label = "Slightly irregular frequency distribution"

    return round(score, 2), label


def analyze_compression_artifacts(image_bytes: bytes) -> Tuple[float, str]:
    """
    Compression consistency analysis.
    Re-saved or composited deepfake images often show double-compression
    artifacts — inconsistent block boundaries across the image.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 50.0, "Cannot decode for compression analysis"

    img_f = img.astype(np.float64)
    h, w = img_f.shape

    # Check 8x8 block boundary discontinuities (JPEG block grid)
    boundary_diffs = []
    for i in range(8, h - 8, 8):
        row_diff = np.abs(img_f[i, :] - img_f[i-1, :]).mean()
        boundary_diffs.append(row_diff)
    for j in range(8, w - 8, 8):
        col_diff = np.abs(img_f[:, j] - img_f[:, j-1]).mean()
        boundary_diffs.append(col_diff)

    if not boundary_diffs:
        return 50.0, "Image too small for block analysis"

    mean_boundary = np.mean(boundary_diffs)
    std_boundary  = np.std(boundary_diffs)
    consistency   = std_boundary / (mean_boundary + 1e-6)

    # Low consistency = uniform blocks = authentic JPEG
    # High consistency = some blocks very different = spliced/double-compressed
    if consistency < 0.5:
        score = 88.0
        label = "Consistent compression — no splicing detected"
    elif consistency < 1.0:
        score = 65.0
        label = "Mild compression inconsistency"
    elif consistency < 2.0:
        score = 42.0
        label = "Moderate inconsistency — possible double-compression"
    else:
        score = 22.0
        label = "Severe compression mismatch — splicing likely"

    return round(score, 2), label


def run_audio_meta_analysis(image_bytes: bytes) -> AudioMetaResult:
    """
    Main entry — accepts raw image bytes.
    Returns AudioMetaResult with hash, EXIF, DCT, compression scores.
    """
    sha256  = compute_sha256(image_bytes)
    exif_score, exif_details = analyze_exif_metadata(image_bytes)
    dct_score, dct_label     = analyze_dct_frequency(image_bytes)
    comp_score, comp_label   = analyze_compression_artifacts(image_bytes)

    # Weighted: EXIF 30%, DCT 40%, Compression 30%
    overall = (
        exif_score * 0.30 +
        dct_score  * 0.40 +
        comp_score * 0.30
    )

    return AudioMetaResult(
        hash_sha256=sha256,
        exif_score=exif_score,
        dct_score=dct_score,
        compression_score=comp_score,
        overall_audio_score=round(overall, 2),
        details={
            "sha256":      sha256,
            "exif":        exif_details,
            "dct":         {"score": dct_score,  "label": dct_label},
            "compression": {"score": comp_score, "label": comp_label},
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# FUTURE EXTENSION: Real audio analysis using librosa
# Uncomment and install: pip install librosa soundfile
# ──────────────────────────────────────────────────────────────────────────────
# import librosa
#
# def analyze_audio_file(audio_path: str) -> dict:
#     y, sr = librosa.load(audio_path, sr=None)
#     mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     spectral  = librosa.feature.spectral_centroid(y=y, sr=sr)
#     zero_cross= librosa.feature.zero_crossing_rate(y)
#     # GAN-synthesized voices show unnaturally consistent MFCCs and
#     # lack the micro-variation present in human speech.
#     mfcc_var  = np.var(mfcc, axis=1).mean()
#     score = min(100, mfcc_var * 5)   # placeholder calibration
#     return {"mfcc_variance": mfcc_var, "score": score}
