from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default (matches the current workspace).
DEFAULT_SONGS_DIR = PROJECT_ROOT / "Songs" / "kpop"

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "tracks.db"

SUPPORTED_AUDIO_EXTENSIONS: tuple[str, ...] = (
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
)


def songs_dir_from_env(default: Path | None = None) -> Path:
    """
    Priority:
      1) env var KPOP_SONGS_DIR
      2) provided default
      3) DEFAULT_SONGS_DIR
    """

    env_path = os.getenv("KPOP_SONGS_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    if default is not None:
        return default
    return DEFAULT_SONGS_DIR.resolve()


def db_path_from_env(default: Path | None = None) -> Path:
    """
    Priority:
      1) env var AI_DJ_DB_PATH
      2) provided default
      3) DEFAULT_DB_PATH
    """

    env_path = os.getenv("AI_DJ_DB_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    if default is not None:
        return default
    return DEFAULT_DB_PATH.resolve()


def is_supported_audio_file(path: Path, supported_exts: Iterable[str] = SUPPORTED_AUDIO_EXTENSIONS) -> bool:
    return path.suffix.lower() in {ext.lower() for ext in supported_exts}

