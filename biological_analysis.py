"""
DeepShield AI — Biological Signals Module
Analyzes facial landmark consistency, skin tone uniformity,
and eye/lip region naturalness as biological authenticity signals.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BiologicalResult:
    skin_tone_score: float        # 0-100
    eye_region_score: float       # 0-100
    landmark_consistency: float   # 0-100
    overall_biological_score: float  # 0-100 (final)
    details: dict


def analyze_skin_tone_uniformity(image_bgr: np.ndarray, face_rect: Optional[tuple]) -> Tuple[float, str]:
    """
    Skin tone analysis within face ROI.
    Deepfakes often have patchy or overly smooth skin tones inconsistent with
    natural lighting variation.
    """
    if face_rect is None:
        return 50.0, "No face region — skin analysis skipped"

    x, y, w, h = face_rect
    face_roi = image_bgr[y:y+h, x:x+w]
    if face_roi.size == 0:
        return 50.0, "Empty face region"

    # Convert to YCrCb for skin segmentation
    ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
    # Typical skin range in YCrCb
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)

    skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
    if skin_ratio < 0.1:
        return 35.0, "Very low skin pixel ratio — unusual face region"

    # Check uniformity: variance of skin pixel intensities
    skin_pixels = face_roi[skin_mask > 0]
    if len(skin_pixels) == 0:
        return 40.0, "No skin pixels detected"

    intensity_std = np.std(skin_pixels.astype(np.float64))

    # Natural skin: moderate variance (not too uniform, not too noisy)
    if 20 <= intensity_std <= 60:
        score = 88.0
        label = "Natural skin tone variation"
    elif intensity_std < 10:
        score = 25.0
        label = "Artificially smooth skin — GAN generation suspected"
    elif intensity_std < 20:
        score = 55.0
        label = "Slightly smooth — mild deepfake risk"
    else:
        score = 50.0
        label = "High variance — possible splicing artifact"

    return round(score, 2), label


def analyze_eye_region(gray: np.ndarray, face_rect: Optional[tuple]) -> Tuple[float, str]:
    """
    Eye region analysis using Haar eye cascade.
    Checks for natural eye detection within face bounds.
    Deepfakes often misplace or blur eye regions.
    """
    if face_rect is None:
        return 50.0, "No face — eye analysis skipped"

    x, y, w, h = face_rect
    # Upper half of face typically contains eyes
    roi_gray = gray[y:y + h//2, x:x+w]

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
    eye_count = len(eyes)

    if eye_count == 2:
        # Check horizontal alignment
        ex1, ey1, ew1, eh1 = eyes[0]
        ex2, ey2, ew2, eh2 = eyes[1]
        center1 = (ex1 + ew1//2, ey1 + eh1//2)
        center2 = (ex2 + ew2//2, ey2 + eh2//2)
        vertical_diff = abs(center1[1] - center2[1])
        eye_spacing = abs(center1[0] - center2[0])

        alignment_ok = vertical_diff < (h * 0.12)  # within 12% height
        spacing_ok   = (w * 0.2) < eye_spacing < (w * 0.7)

        if alignment_ok and spacing_ok:
            score = 92.0
            label = "2 eyes detected — natural alignment"
        elif alignment_ok:
            score = 70.0
            label = "Eyes aligned but unusual spacing"
        else:
            score = 45.0
            label = "Misaligned eyes — possible deepfake"

    elif eye_count == 1:
        score = 55.0
        label = "Only 1 eye detected — partial occlusion or deepfake"
    elif eye_count == 0:
        score = 30.0
        label = "No eyes detected — strong deepfake or occlusion"
    else:
        score = 50.0
        label = f"{eye_count} eye-like regions — unusual"

    return round(score, 2), label


def analyze_landmark_consistency(gray: np.ndarray, face_rect: Optional[tuple]) -> Tuple[float, str]:
    """
    Approximate landmark consistency using gradient-based facial region checks.
    Assesses whether the face interior has expected structural gradients.
    """
    if face_rect is None:
        return 50.0, "No face — landmark check skipped"

    x, y, w, h = face_rect
    face_gray = gray[y:y+h, x:x+w]

    if face_gray.size == 0:
        return 50.0, "Empty face region"

    # Sobel gradients — natural faces have well-structured gradient fields
    sobelx = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Natural face: gradient concentrated in feature regions (eyes, nose, mouth)
    # GAN faces often have diffuse, uniform gradients
    grad_mean = magnitude.mean()
    grad_std  = magnitude.std()
    concentration = grad_std / (grad_mean + 1e-6)  # higher = more structured

    if concentration > 1.5:
        score = 85.0
        label = "Structured gradient — natural facial landmarks"
    elif concentration > 0.8:
        score = 65.0
        label = "Moderate structure — some inconsistency"
    else:
        score = 35.0
        label = "Diffuse gradient — GAN/deepfake structure suspected"

    return round(score, 2), label


def run_biological_analysis(image_bytes: bytes) -> BiologicalResult:
    """
    Main entry — accepts raw image bytes.
    Returns BiologicalResult with individual and overall biological scores.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image.")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detect face
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    face_rect = None
    if len(faces) > 0:
        face_rect = tuple(sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0])

    skin_score,  skin_label  = analyze_skin_tone_uniformity(img_bgr, face_rect)
    eye_score,   eye_label   = analyze_eye_region(gray, face_rect)
    lm_score,    lm_label    = analyze_landmark_consistency(gray, face_rect)

    # Weighted: skin 35%, eyes 35%, landmarks 30%
    overall = (
        skin_score * 0.35 +
        eye_score  * 0.35 +
        lm_score   * 0.30
    )

    return BiologicalResult(
        skin_tone_score=skin_score,
        eye_region_score=eye_score,
        landmark_consistency=lm_score,
        overall_biological_score=round(overall, 2),
        details={
            "skin_tone":  {"score": skin_score, "label": skin_label},
            "eye_region": {"score": eye_score,  "label": eye_label},
            "landmarks":  {"score": lm_score,   "label": lm_label},
            "face_found": face_rect is not None,
        }
    )
