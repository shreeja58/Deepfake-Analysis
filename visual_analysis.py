"""
DeepShield AI — Visual Analysis Module
Detects deepfake indicators using OpenCV-based image processing.
Techniques: Laplacian blur, Canny edge, noise variance, face detection.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class VisualResult:
    blur_score: float          # 0-100 (higher = sharper / more authentic)
    edge_score: float          # 0-100
    noise_score: float         # 0-100
    face_detected: bool
    face_count: int
    face_symmetry_score: float # 0-100
    overall_visual_score: float  # 0-100 (final weighted visual score)
    details: dict


def analyze_blur(gray: np.ndarray) -> Tuple[float, str]:
    """
    Laplacian variance — measures image sharpness.
    Deepfakes often have unnatural blur patterns at face boundaries.
    Returns score 0-100 (100 = perfectly sharp / authentic).
    """
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Normalize: typical authentic images > 100, blurry/deepfake < 30
    if laplacian_var >= 200:
        score = 100.0
        label = "Sharp — no blur artifacts"
    elif laplacian_var >= 100:
        score = 80.0 + (laplacian_var - 100) / 10
        label = "Slightly soft — minor blur detected"
    elif laplacian_var >= 40:
        score = 40.0 + (laplacian_var - 40) * (40 / 60)
        label = "Moderate blur — suspicious"
    else:
        score = max(0, laplacian_var)
        label = "Heavy blur — deepfake artifact likely"

    score = min(100.0, score)
    return round(score, 2), label, round(laplacian_var, 2)


def analyze_edges(gray: np.ndarray) -> Tuple[float, str]:
    """
    Canny edge analysis — checks edge consistency.
    Composited deepfakes show inconsistent edge density at splice regions.
    """
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size  # ratio of edge pixels

    # Natural images typically 3-15% edge density
    if 0.03 <= edge_density <= 0.15:
        score = 90.0
        label = "Natural edge distribution"
    elif 0.015 <= edge_density < 0.03 or 0.15 < edge_density <= 0.25:
        score = 65.0
        label = "Slightly abnormal edge density"
    elif edge_density < 0.015:
        score = 35.0
        label = "Too few edges — possible AI generation"
    else:
        score = 30.0
        label = "Excessive edges — compression/GAN artifact"

    return round(score, 2), label, round(edge_density * 100, 3)


def analyze_noise(gray: np.ndarray) -> Tuple[float, str]:
    """
    Statistical noise analysis.
    Authentic camera images have natural sensor noise.
    AI-generated / deepfake content has unnatural noise patterns.
    """
    # High-pass filter to isolate noise
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    noise_map = np.abs(gray.astype(np.float32) - blurred)
    noise_std = np.std(noise_map)

    # Also compute local variance uniformity
    local_var = []
    h, w = gray.shape
    step = max(1, min(h, w) // 8)
    for i in range(0, h - step, step):
        for j in range(0, w - step, step):
            patch = gray[i:i+step, j:j+step]
            local_var.append(np.var(patch.astype(np.float64)))
    var_uniformity = np.std(local_var) / (np.mean(local_var) + 1e-6)

    # Natural images: moderate noise_std (2-8), non-uniform local variance
    if 2 <= noise_std <= 10 and var_uniformity > 0.5:
        score = 88.0
        label = "Natural sensor noise pattern"
    elif noise_std < 1.5:
        score = 30.0
        label = "Abnormally clean — AI generation suspected"
    elif noise_std > 15:
        score = 40.0
        label = "Excessive noise — heavy compression or splicing"
    else:
        score = 60.0
        label = "Mildly irregular noise pattern"

    return round(score, 2), label, round(float(noise_std), 3)


def analyze_faces(image_bgr: np.ndarray) -> Tuple[bool, int, float, str]:
    """
    Face detection + symmetry analysis using OpenCV Haar cascades.
    Deepfakes often show asymmetric face regions or impossible geometry.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_count = len(faces)

    if face_count == 0:
        return False, 0, 50.0, "No face detected — cannot verify facial consistency"

    # Analyze first/largest face for symmetry
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    face_roi = gray[y:y+h, x:x+w]

    if face_roi.size == 0:
        return True, face_count, 50.0, "Face found but ROI invalid"

    # Symmetry: compare left vs right half
    mid = face_roi.shape[1] // 2
    left_half = face_roi[:, :mid].astype(np.float64)
    right_half = cv2.flip(face_roi[:, mid:mid + mid], 1).astype(np.float64)

    if left_half.shape == right_half.shape:
        diff = np.abs(left_half - right_half)
        symmetry_ratio = 1 - (diff.mean() / 255)
        symmetry_score = round(symmetry_ratio * 100, 2)
    else:
        symmetry_score = 50.0

    if symmetry_score >= 75:
        label = f"{face_count} face(s) detected — symmetry normal"
    elif symmetry_score >= 55:
        label = f"{face_count} face(s) detected — mild asymmetry"
    else:
        label = f"{face_count} face(s) detected — high asymmetry (deepfake risk)"

    return True, face_count, symmetry_score, label


def run_visual_analysis(image_bytes: bytes) -> VisualResult:
    """
    Main entry point — accepts raw image bytes.
    Returns a VisualResult with all scores and a final overall visual score.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image. Ensure it is a valid JPG/PNG.")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blur_score, blur_label, blur_raw = analyze_blur(gray)
    edge_score, edge_label, edge_raw = analyze_edges(gray)
    noise_score, noise_label, noise_raw = analyze_noise(gray)
    face_detected, face_count, symmetry_score, face_label = analyze_faces(img_bgr)

    # Weighted final visual score
    # Blur: 35%, Edge: 25%, Noise: 25%, Face symmetry: 15%
    if face_detected:
        overall = (
            blur_score   * 0.35 +
            edge_score   * 0.25 +
            noise_score  * 0.25 +
            symmetry_score * 0.15
        )
    else:
        # No face — redistribute face weight
        overall = (
            blur_score  * 0.40 +
            edge_score  * 0.30 +
            noise_score * 0.30
        )

    return VisualResult(
        blur_score=blur_score,
        edge_score=edge_score,
        noise_score=noise_score,
        face_detected=face_detected,
        face_count=face_count,
        face_symmetry_score=symmetry_score,
        overall_visual_score=round(overall, 2),
        details={
            "blur":  {"score": blur_score,  "label": blur_label,  "raw_variance": blur_raw},
            "edges": {"score": edge_score,  "label": edge_label,  "edge_density_%": edge_raw},
            "noise": {"score": noise_score, "label": noise_label, "noise_std": noise_raw},
            "face":  {"detected": face_detected, "count": face_count,
                      "symmetry_score": symmetry_score, "label": face_label},
        }
    )
