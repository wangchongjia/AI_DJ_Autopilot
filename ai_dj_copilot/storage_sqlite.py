from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

import pandas as pd

from .models import TrackFeatures


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tracks (
  file_path TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  artist TEXT NOT NULL,

  duration_sec REAL NOT NULL,
  bpm REAL NULL,
  estimated_key TEXT NULL,

  rms_mean REAL NOT NULL,
  loudness_db_mean REAL NOT NULL,
  energy_score REAL NOT NULL,

  spectral_centroid_mean_hz REAL NOT NULL,
  zero_crossing_rate_mean REAL NOT NULL,

  segment_window_sec REAL NOT NULL DEFAULT 15,
  outro_rms_mean REAL NOT NULL DEFAULT 0,
  outro_loudness_db_mean REAL NOT NULL DEFAULT -80,
  outro_silence_ratio REAL NOT NULL DEFAULT 0,
  outro_silence_at_end_flag INTEGER NOT NULL DEFAULT 0,
  intro_rms_mean REAL NOT NULL DEFAULT 0,
  intro_loudness_db_mean REAL NOT NULL DEFAULT -80,
  intro_onset_strength_mean REAL NOT NULL DEFAULT 0,
  outro_energy_score REAL NOT NULL DEFAULT 0,
  intro_energy_score REAL NOT NULL DEFAULT 0,

  -- ISO-8601 timestamp (UTC-ish) updated on every ingest upsert
  updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

-- Directed transition labels between tracks (A -> B).
-- label: 1 compatible, 0 incompatible, NULL = skip.
CREATE TABLE IF NOT EXISTS transition_labels (
  track_a_file_path TEXT NOT NULL,
  track_b_file_path TEXT NOT NULL,
  label INTEGER NULL,

  created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),

  PRIMARY KEY (track_a_file_path, track_b_file_path),
  CHECK (label IS NULL OR label IN (0, 1))
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_tracks_columns(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(tracks);")
    existing = {row[1] for row in cur.fetchall()}
    alters: list[tuple[str, str]] = [
        ("segment_window_sec", "REAL NOT NULL DEFAULT 15"),
        ("outro_rms_mean", "REAL NOT NULL DEFAULT 0"),
        ("outro_loudness_db_mean", "REAL NOT NULL DEFAULT -80"),
        ("outro_silence_ratio", "REAL NOT NULL DEFAULT 0"),
        ("outro_silence_at_end_flag", "INTEGER NOT NULL DEFAULT 0"),
        ("intro_rms_mean", "REAL NOT NULL DEFAULT 0"),
        ("intro_loudness_db_mean", "REAL NOT NULL DEFAULT -80"),
        ("intro_onset_strength_mean", "REAL NOT NULL DEFAULT 0"),
        ("outro_energy_score", "REAL NOT NULL DEFAULT 0"),
        ("intro_energy_score", "REAL NOT NULL DEFAULT 0"),
    ]
    for col, decl in alters:
        if col not in existing:
            conn.execute(f"ALTER TABLE tracks ADD COLUMN {col} {decl};")


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    _migrate_tracks_columns(conn)
    conn.commit()


def upsert_tracks(conn: sqlite3.Connection, tracks: Sequence[TrackFeatures]) -> None:
    sql = """
    INSERT INTO tracks (
      file_path, title, artist, duration_sec, bpm, estimated_key,
      rms_mean, loudness_db_mean, energy_score,
      spectral_centroid_mean_hz, zero_crossing_rate_mean,
      segment_window_sec, outro_rms_mean, outro_loudness_db_mean, outro_silence_ratio,
      outro_silence_at_end_flag, intro_rms_mean, intro_loudness_db_mean, intro_onset_strength_mean,
      outro_energy_score, intro_energy_score,
      updated_at
    ) VALUES (
      :file_path, :title, :artist, :duration_sec, :bpm, :estimated_key,
      :rms_mean, :loudness_db_mean, :energy_score,
      :spectral_centroid_mean_hz, :zero_crossing_rate_mean,
      :segment_window_sec, :outro_rms_mean, :outro_loudness_db_mean, :outro_silence_ratio,
      :outro_silence_at_end_flag, :intro_rms_mean, :intro_loudness_db_mean, :intro_onset_strength_mean,
      :outro_energy_score, :intro_energy_score,
      CURRENT_TIMESTAMP
    )
    ON CONFLICT(file_path) DO UPDATE SET
      title=excluded.title,
      artist=excluded.artist,
      duration_sec=excluded.duration_sec,
      bpm=excluded.bpm,
      estimated_key=excluded.estimated_key,
      rms_mean=excluded.rms_mean,
      loudness_db_mean=excluded.loudness_db_mean,
      energy_score=excluded.energy_score,
      spectral_centroid_mean_hz=excluded.spectral_centroid_mean_hz,
      zero_crossing_rate_mean=excluded.zero_crossing_rate_mean,
      segment_window_sec=excluded.segment_window_sec,
      outro_rms_mean=excluded.outro_rms_mean,
      outro_loudness_db_mean=excluded.outro_loudness_db_mean,
      outro_silence_ratio=excluded.outro_silence_ratio,
      outro_silence_at_end_flag=excluded.outro_silence_at_end_flag,
      intro_rms_mean=excluded.intro_rms_mean,
      intro_loudness_db_mean=excluded.intro_loudness_db_mean,
      intro_onset_strength_mean=excluded.intro_onset_strength_mean,
      outro_energy_score=excluded.outro_energy_score,
      intro_energy_score=excluded.intro_energy_score,
      updated_at=CURRENT_TIMESTAMP;
    """

    # SQLite uses named parameters; dataclass has matching attribute names.
    conn.executemany(sql, (t.__dict__ for t in tracks))
    conn.commit()


def fetch_all_tracks(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM tracks ORDER BY title ASC;", conn)
    return df


def fetch_transition_labels(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Returns labeled pairs only.
    Note: label can be NULL (skip), which still counts as an answered pair.
    """
    df = pd.read_sql_query(
        """
        SELECT track_a_file_path, track_b_file_path, label, created_at, updated_at
        FROM transition_labels;
        """,
        conn,
    )
    return df


def upsert_transition_label(
    conn: sqlite3.Connection,
    *,
    track_a_file_path: str,
    track_b_file_path: str,
    label: int | None,
) -> None:
    sql = """
    INSERT INTO transition_labels (
      track_a_file_path, track_b_file_path, label,
      created_at, updated_at
    ) VALUES (
      :track_a_file_path, :track_b_file_path, :label,
      CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
    )
    ON CONFLICT(track_a_file_path, track_b_file_path) DO UPDATE SET
      label=excluded.label,
      updated_at=CURRENT_TIMESTAMP;
    """
    conn.execute(
        sql,
        {
            "track_a_file_path": track_a_file_path,
            "track_b_file_path": track_b_file_path,
            "label": label,
        },
    )
    conn.commit()

