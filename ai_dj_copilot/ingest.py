from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .audio_features import extract_track_features
from .config import SUPPORTED_AUDIO_EXTENSIONS, is_supported_audio_file
from .models import TrackFeatures
from .storage_sqlite import connect, init_db, upsert_tracks


def scan_audio_files(songs_dir: Path, *, supported_exts: Iterable[str] = SUPPORTED_AUDIO_EXTENSIONS) -> list[Path]:
    if not songs_dir.exists():
        raise FileNotFoundError(f"songs_dir does not exist: {songs_dir}")
    if not songs_dir.is_dir():
        raise NotADirectoryError(f"songs_dir is not a directory: {songs_dir}")

    files: list[Path] = []
    for path in songs_dir.rglob("*"):
        if path.is_file() and is_supported_audio_file(path, supported_exts=supported_exts):
            files.append(path)
    return sorted(files)


def normalize_energy_scores(tracks: Sequence[TrackFeatures]) -> list[TrackFeatures]:
    rms_values = [t.rms_mean for t in tracks]
    rms_min = float(min(rms_values)) if rms_values else 0.0
    rms_max = float(max(rms_values)) if rms_values else 0.0
    denom = rms_max - rms_min

    def _norm(v: float) -> float:
        if abs(denom) < 1e-12:
            # All tracks have the same RMS (unlikely, but keep stable).
            return 0.5
        return (v - rms_min) / denom

    normalized: list[TrackFeatures] = []
    for t in tracks:
        normalized.append(replace(t, energy_score=_norm(t.rms_mean)))
    return normalized


def normalize_segment_energy_scores(tracks: Sequence[TrackFeatures]) -> list[TrackFeatures]:
    """
    Per-batch min/max normalization of outro/intro RMS into 0..1 scores (local library relative).
    """
    if not tracks:
        return []

    outro_vals = [t.outro_rms_mean for t in tracks]
    intro_vals = [t.intro_rms_mean for t in tracks]
    omin, omax = float(min(outro_vals)), float(max(outro_vals))
    imin, imax = float(min(intro_vals)), float(max(intro_vals))

    def _norm(v: float, lo: float, hi: float) -> float:
        d = hi - lo
        if abs(d) < 1e-12:
            return 0.5
        return (v - lo) / d

    out: list[TrackFeatures] = []
    for t in tracks:
        out.append(
            replace(
                t,
                outro_energy_score=_norm(t.outro_rms_mean, omin, omax),
                intro_energy_score=_norm(t.intro_rms_mean, imin, imax),
            )
        )
    return out


def ingest_library(
    *,
    songs_dir: Path,
    db_path: Path,
    target_sr: int = 22050,
    segment_window_sec: float = 15.0,
    supported_exts: Iterable[str] = SUPPORTED_AUDIO_EXTENSIONS,
) -> int:
    """
    Prototype ingestion:
    - Extract features for all supported audio files under songs_dir
    - Normalize energy_score across the full extracted set
    - Upsert into SQLite
    """

    audio_files = scan_audio_files(songs_dir, supported_exts=supported_exts)
    if not audio_files:
        print(f"[ingest] No audio files found under: {songs_dir}")
        return 0

    print(f"[ingest] Found {len(audio_files)} audio files. Extracting features...")
    extracted: list[TrackFeatures] = []

    for i, path in enumerate(audio_files, start=1):
        try:
            print(f"[ingest] ({i}/{len(audio_files)}) Extract: {path.name}")
            features = extract_track_features(path, target_sr=target_sr, segment_window_sec=segment_window_sec)
            extracted.append(features)
        except Exception as e:
            # Keep ingestion resilient for a local prototype.
            print(f"[ingest] WARNING: failed to process {path}: {e}")

    if not extracted:
        print("[ingest] No tracks were successfully extracted.")
        return 0

    normalized_tracks = normalize_energy_scores(extracted)
    normalized_tracks = normalize_segment_energy_scores(normalized_tracks)
    conn = connect(db_path)
    try:
        init_db(conn)
        upsert_tracks(conn, normalized_tracks)
    finally:
        conn.close()

    print(f"[ingest] Done. Stored {len(normalized_tracks)} tracks into {db_path}")
    return len(normalized_tracks)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest a local K-pop audio library into SQLite.")
    parser.add_argument("--songs-dir", type=str, default=None, help="Directory to scan for audio files.")
    parser.add_argument("--db", type=str, default=None, help="SQLite db path (e.g. data/tracks.db).")
    parser.add_argument("--target-sr", type=int, default=22050, help="Resample rate for librosa.")
    parser.add_argument(
        "--segment-window",
        type=float,
        default=15.0,
        help="Seconds for intro/outro windows (10–20 typical).",
    )
    args = parser.parse_args(argv)

    from .config import DEFAULT_SONGS_DIR, db_path_from_env, songs_dir_from_env

    songs_dir = Path(args.songs_dir).expanduser().resolve() if args.songs_dir else songs_dir_from_env(DEFAULT_SONGS_DIR)
    db_path = Path(args.db).expanduser().resolve() if args.db else db_path_from_env()

    ingest_library(
        songs_dir=songs_dir,
        db_path=db_path,
        target_sr=args.target_sr,
        segment_window_sec=float(args.segment_window),
    )


if __name__ == "__main__":
    main()

