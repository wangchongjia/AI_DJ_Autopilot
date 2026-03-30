from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrackFeatures:
    file_path: str
    title: str
    artist: str

    # Time domain
    duration_sec: float

    # Rhythm / musicality
    bpm: Optional[float]
    estimated_key: Optional[str]

    # Loudness / energy
    rms_mean: float
    loudness_db_mean: float
    energy_score: float  # 0..1 (normalized across the whole library run)

    # Spectral / signal proxies
    spectral_centroid_mean_hz: float
    zero_crossing_rate_mean: float

    # Mix-boundary segments (same window length for intro/outro; see audio_features)
    segment_window_sec: float
    outro_rms_mean: float
    outro_loudness_db_mean: float
    outro_silence_ratio: float
    outro_silence_at_end_flag: int  # 0/1: likely hard stop / long silence at very end
    intro_rms_mean: float
    intro_loudness_db_mean: float
    intro_onset_strength_mean: float
    outro_energy_score: float  # 0..1 from batch-normalized outro RMS
    intro_energy_score: float  # 0..1 from batch-normalized intro RMS

