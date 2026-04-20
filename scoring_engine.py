"""
DeepShield AI — Ensemble Scoring Engine
Combines Visual (40%), Biological (35%), and Audio/Meta (25%) scores
into a final confidence value with verdict classification.

Formula: Confidence = 0.40 × V + 0.35 × B + 0.25 × A
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DeepShieldVerdict:
    visual_score: float
    biological_score: float
    audio_score: float
    confidence: float          # 0-100 final score
    verdict: str               # "AUTHENTIC" | "SUSPICIOUS" | "DEEPFAKE"
    verdict_color: str         # for UI rendering
    verdict_emoji: str
    risk_level: str
    recommendation: str
    breakdown: dict


WEIGHTS = {
    "visual":     0.40,
    "biological": 0.35,
    "audio":      0.25,
}

THRESHOLDS = {
    "authentic":  70,
    "suspicious": 45,
    # below 45 = DEEPFAKE
}


def classify_verdict(confidence: float) -> Tuple[str, str, str, str, str]:
    """Return verdict string, color, emoji, risk level, and recommendation."""
    if confidence >= THRESHOLDS["authentic"]:
        return (
            "AUTHENTIC",
            "#00C853",
            "✅",
            "LOW RISK",
            "Content appears genuine. Safe to publish or share."
        )
    elif confidence >= THRESHOLDS["suspicious"]:
        return (
            "SUSPICIOUS",
            "#FF9100",
            "⚠️",
            "MEDIUM RISK",
            "Possible manipulation detected. Request source verification before publishing."
        )
    else:
        return (
            "DEEPFAKE",
            "#FF1744",
            "🚨",
            "HIGH RISK",
            "High probability of AI-generated or manipulated content. Do NOT publish."
        )


def compute_confidence(
    visual_score: float,
    biological_score: float,
    audio_score: float
) -> DeepShieldVerdict:
    """
    Compute ensemble confidence and return a full verdict object.

    Args:
        visual_score:     0-100 from visual_analysis module
        biological_score: 0-100 from biological_analysis module
        audio_score:      0-100 from audio_meta_analysis module

    Returns:
        DeepShieldVerdict with all scoring details and classification.
    """
    # Clamp inputs
    v = max(0.0, min(100.0, visual_score))
    b = max(0.0, min(100.0, biological_score))
    a = max(0.0, min(100.0, audio_score))

    confidence = (
        v * WEIGHTS["visual"]     +
        b * WEIGHTS["biological"] +
        a * WEIGHTS["audio"]
    )
    confidence = round(confidence, 2)

    verdict, color, emoji, risk, recommendation = classify_verdict(confidence)

    # Per-module verdict labels
    def score_label(s):
        if s >= 70: return "✅ Strong"
        if s >= 45: return "⚠️ Moderate"
        return "🚨 Weak"

    return DeepShieldVerdict(
        visual_score=v,
        biological_score=b,
        audio_score=a,
        confidence=confidence,
        verdict=verdict,
        verdict_color=color,
        verdict_emoji=emoji,
        risk_level=risk,
        recommendation=recommendation,
        breakdown={
            "formula": "0.40 × Visual + 0.35 × Biological + 0.25 × Audio/Meta",
            "weights": WEIGHTS,
            "thresholds": THRESHOLDS,
            "contributions": {
                "visual":     round(v * WEIGHTS["visual"], 2),
                "biological": round(b * WEIGHTS["biological"], 2),
                "audio":      round(a * WEIGHTS["audio"], 2),
            },
            "module_labels": {
                "visual":     score_label(v),
                "biological": score_label(b),
                "audio":      score_label(a),
            }
        }
    )
