from __future__ import annotations

from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from .models import TrackFeatures


NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Last N seconds ≈ outro; first N seconds ≈ intro (prototype; not full cue-point search).
DEFAULT_SEGMENT_WINDOW_SEC = 15.0


def _infer_title_from_filename(file_path: Path) -> str:
    # Prototype heuristic: file stem as title, strip common suffixes.
    stem = file_path.stem
    # e.g. "Artist - Title (1)" -> "Title (1)" (still simple)
    parts = stem.split(" - ", maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    return stem.strip()


def _infer_artist_from_filename(file_path: Path) -> str:
    stem = file_path.stem
    parts = stem.split(" - ", maxsplit=1)
    if len(parts) == 2:
        # Prototype: often "Artist - Title.ext"
        return parts[0].strip()
    return ""


def _infer_key_from_chroma(chroma_mean: np.ndarray) -> Optional[str]:
    """
    Simple key estimation by correlating average chroma with Krumhansl-Schmuckler profiles.

    Reliability note:
    - This is a heuristic for a prototype and may be inaccurate for short/heterogeneous tracks.
    - If you later want accurate keys, you will likely need better algorithms or dataset-specific handling.
    """

    if chroma_mean.shape != (12,):
        return None

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    # Normalize profiles and chroma to improve numeric stability.
    major_profile = major_profile / (np.linalg.norm(major_profile) + 1e-12)
    minor_profile = minor_profile / (np.linalg.norm(minor_profile) + 1e-12)

    chroma = chroma_mean.astype(np.float64)
    chroma = chroma / (np.linalg.norm(chroma) + 1e-12)

    def _cosine_sim(template: np.ndarray, shift: int) -> float:
        rolled = np.roll(template, shift)
        return float(np.dot(chroma, rolled))

    best_major = max(range(12), key=lambda i: _cosine_sim(major_profile, i))
    best_minor = max(range(12), key=lambda i: _cosine_sim(minor_profile, i))

    # Choose the best match among major/minor (simple comparison).
    major_score = _cosine_sim(major_profile, best_major)
    minor_score = _cosine_sim(minor_profile, best_minor)

    if major_score >= minor_score:
        return f"{NOTE_NAMES_SHARP[best_major]} major"
    return f"{NOTE_NAMES_SHARP[best_minor]} minor"


def estimate_tempo_bpm(y: np.ndarray, sr: int) -> Optional[float]:
    """
    Prototype BPM estimate using onset strength envelope + librosa.beat.tempo.
    """

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempi = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    if tempi.size == 0:
        return None
    # tempi[0] is typically the highest-confidence tempo.
    return float(tempi[0])


def _segment_sample_ranges(
    n_samples: int,
    sr: int,
    *,
    window_sec: float,
) -> tuple[int, int, int, int]:
    """
    Return (intro_start, intro_end, outro_start, outro_end) sample indices.
    Windows are capped so intro and outro do not overlap (short tracks).
    """
    if n_samples <= 0 or sr <= 0:
        return 0, 0, 0, 0
    win = int(window_sec * sr)
    win = max(1, min(win, n_samples // 2))
    intro_start = 0
    intro_end = min(win, n_samples)
    outro_end = n_samples
    outro_start = max(0, n_samples - win)
    if outro_start < intro_end:
        mid = n_samples // 2
        intro_end = mid
        outro_start = mid
    return intro_start, intro_end, outro_start, outro_end


def extract_mix_boundary_segment_features(
    y: np.ndarray,
    sr: int,
    *,
    window_sec: float = DEFAULT_SEGMENT_WINDOW_SEC,
) -> dict[str, float]:
    """
    Section-aware features for likely mix boundaries (prototype).

    - Outro: last `window_sec` seconds (or shorter if track is brief).
    - Intro: first `window_sec` seconds (or shorter if track is brief).

    outro_silence_ratio: fraction of samples in the outro window whose |amplitude| is below a
    small relative threshold vs global peak (rough silence / low-energy proxy).

    intro_onset_strength_mean: mean librosa onset strength envelope over the intro window
    (higher ≈ stronger attacks / more transient energy at the start).

    outro_silence_at_end_flag: 1 if the last ~2s of the *full* track is mostly silent
    (optional hard-stop / long fade tail heuristic; unreliable on heavily compressed masters).
    """
    n = y.size
    if n == 0 or sr <= 0:
        return {
            "outro_rms_mean": 0.0,
            "outro_loudness_db_mean": -np.inf,
            "outro_silence_ratio": 0.0,
            "intro_rms_mean": 0.0,
            "intro_loudness_db_mean": -np.inf,
            "intro_onset_strength_mean": 0.0,
            "outro_silence_at_end_flag": 0,
        }

    intro_s, intro_e, outro_s, outro_e = _segment_sample_ranges(n, sr, window_sec=window_sec)
    y_intro = y[intro_s:intro_e]
    y_outro = y[outro_s:outro_e]

    eps = 1e-10
    peak = float(np.max(np.abs(y)) + eps)
    thr = 0.01 * peak  # 1% of peak — simple relative silence threshold

    def _rms(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

    def _loud_db(rms_v: float) -> float:
        return float(20.0 * np.log10(rms_v + eps))

    outro_rms = _rms(y_outro)
    intro_rms = _rms(y_intro)
    outro_silence_ratio = float(np.mean(np.abs(y_outro) < thr)) if y_outro.size else 0.0

    if y_intro.size > 0:
        onset_intro = librosa.onset.onset_strength(y=y_intro, sr=sr)
        intro_onset = float(np.mean(onset_intro)) if onset_intro.size else 0.0
    else:
        intro_onset = 0.0

    # Very end of track (last ~2s): silence-at-end flag
    tail_n = min(int(2.0 * sr), n)
    y_tail = y[-tail_n:]
    tail_silence_ratio = float(np.mean(np.abs(y_tail) < thr)) if y_tail.size else 0.0
    outro_silence_at_end_flag = int(tail_silence_ratio >= 0.5)

    return {
        "outro_rms_mean": outro_rms,
        "outro_loudness_db_mean": _loud_db(outro_rms),
        "outro_silence_ratio": outro_silence_ratio,
        "intro_rms_mean": intro_rms,
        "intro_loudness_db_mean": _loud_db(intro_rms),
        "intro_onset_strength_mean": intro_onset,
        "outro_silence_at_end_flag": int(outro_silence_at_end_flag),
    }


def extract_track_features(
    file_path: Path,
    *,
    target_sr: int = 22050,
    segment_window_sec: float = DEFAULT_SEGMENT_WINDOW_SEC,
) -> TrackFeatures:
    """
    Extract track-level audio features.

    energy_score is set to 0 here; it will be normalized across the library during ingestion.
    """

    title = _infer_title_from_filename(file_path)
    artist = _infer_artist_from_filename(file_path)

    # librosa will rely on soundfile/audioread backends. For mp3, you may need system ffmpeg.
    y, sr = librosa.load(str(file_path), sr=target_sr, mono=True)

    duration_sec = float(len(y) / sr) if sr > 0 else 0.0
    if y.size == 0:
        # Should rarely happen; keep values safe for prototype.
        return TrackFeatures(
            file_path=str(file_path),
            title=title,
            artist=artist,
            duration_sec=duration_sec,
            bpm=None,
            estimated_key=None,
            rms_mean=0.0,
            loudness_db_mean=-np.inf,
            energy_score=0.0,
            spectral_centroid_mean_hz=0.0,
            zero_crossing_rate_mean=0.0,
            segment_window_sec=float(segment_window_sec),
            outro_rms_mean=0.0,
            outro_loudness_db_mean=-np.inf,
            outro_silence_ratio=0.0,
            outro_silence_at_end_flag=0,
            intro_rms_mean=0.0,
            intro_loudness_db_mean=-np.inf,
            intro_onset_strength_mean=0.0,
            outro_energy_score=0.0,
            intro_energy_score=0.0,
        )

    # Loudness/energy proxy
    rms_frames = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms_frames))

    eps = 1e-10
    loudness_db_frames = 20.0 * np.log10(rms_frames + eps)
    loudness_db_mean = float(np.mean(loudness_db_frames))

    # Spectral / signal proxies
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_centroid_mean_hz = float(np.mean(centroid))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zero_crossing_rate_mean = float(np.mean(zcr))

    # Tempo (BPM)
    bpm = estimate_tempo_bpm(y, sr)

    # Key (estimated)
    # Using average chroma as a simplified proxy.
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    estimated_key = _infer_key_from_chroma(chroma_mean)

    seg = extract_mix_boundary_segment_features(y, sr, window_sec=segment_window_sec)

    return TrackFeatures(
        file_path=str(file_path),
        title=title,
        artist=artist,
        duration_sec=duration_sec,
        bpm=bpm,
        estimated_key=estimated_key,
        rms_mean=rms_mean,
        loudness_db_mean=loudness_db_mean,
        energy_score=0.0,
        spectral_centroid_mean_hz=spectral_centroid_mean_hz,
        zero_crossing_rate_mean=zero_crossing_rate_mean,
        segment_window_sec=float(segment_window_sec),
        outro_rms_mean=float(seg["outro_rms_mean"]),
        outro_loudness_db_mean=float(seg["outro_loudness_db_mean"]),
        outro_silence_ratio=float(seg["outro_silence_ratio"]),
        outro_silence_at_end_flag=int(seg["outro_silence_at_end_flag"]),
        intro_rms_mean=float(seg["intro_rms_mean"]),
        intro_loudness_db_mean=float(seg["intro_loudness_db_mean"]),
        intro_onset_strength_mean=float(seg["intro_onset_strength_mean"]),
        outro_energy_score=0.0,
        intro_energy_score=0.0,
    )

