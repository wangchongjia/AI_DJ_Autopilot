from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

# Ensure local package imports work when running `python scripts/...py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_dj_copilot.config import db_path_from_env
from ai_dj_copilot.storage_sqlite import connect, init_db


def _load_paths_from_file(paths_file: Optional[str]) -> list[str]:
    if not paths_file:
        return []
    p = Path(paths_file).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"paths file not found: {p}")
    out: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        v = line.strip()
        if not v or v.startswith("#"):
            continue
        out.append(v)
    return out


def _fetch_candidates(
    conn,
    *,
    duration_lt: Optional[float],
    remove_paths: Sequence[str],
) -> list[str]:
    remove_set: set[str] = set(remove_paths)
    if duration_lt is not None:
        rows = conn.execute(
            "SELECT file_path FROM tracks WHERE duration_sec < ?",
            (float(duration_lt),),
        ).fetchall()
        remove_set.update(str(r[0]) for r in rows)
    return sorted(remove_set)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cleanup tracks in SQLite and cascade-delete related transition labels."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite path; default uses AI_DJ_DB_PATH or data/tracks.db",
    )
    parser.add_argument(
        "--duration-lt",
        type=float,
        default=None,
        help="Remove tracks where duration_sec < this value (e.g. 90).",
    )
    parser.add_argument(
        "--file-path",
        action="append",
        default=[],
        help="Exact track file_path to remove. Can be repeated.",
    )
    parser.add_argument(
        "--paths-file",
        type=str,
        default=None,
        help="Optional text file containing file_path values to remove (one per line).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview matched rows only; no DB changes.",
    )
    args = parser.parse_args(argv)

    remove_paths = list(args.file_path or [])
    remove_paths.extend(_load_paths_from_file(args.paths_file))
    remove_paths = sorted(set(remove_paths))

    if args.duration_lt is None and not remove_paths:
        raise SystemExit("No cleanup criteria provided. Use --duration-lt and/or --file-path/--paths-file.")

    db_path = Path(args.db).expanduser().resolve() if args.db else db_path_from_env()
    conn = connect(db_path)
    try:
        init_db(conn)
        candidates = _fetch_candidates(
            conn,
            duration_lt=args.duration_lt,
            remove_paths=remove_paths,
        )

        if not candidates:
            print("[cleanup_tracks] No tracks matched. Nothing to do.")
            print("  tracks_matched: 0")
            print("  tracks_removed: 0")
            print("  labels_removed: 0")
            return

        placeholders = ",".join(["?"] * len(candidates))
        labels_to_remove = conn.execute(
            f"""
            SELECT COUNT(*) FROM transition_labels
            WHERE track_a_file_path IN ({placeholders})
               OR track_b_file_path IN ({placeholders})
            """,
            tuple(candidates + candidates),
        ).fetchone()[0]

        print(f"[cleanup_tracks] tracks_matched: {len(candidates)}")
        print(f"[cleanup_tracks] labels_that_would_be_removed: {labels_to_remove}")
        print("[cleanup_tracks] sample matched tracks:")
        for p in candidates[:20]:
            print(f"  - {p}")
        if len(candidates) > 20:
            print(f"  ... and {len(candidates) - 20} more")

        if args.dry_run:
            print("[cleanup_tracks] DRY RUN: no rows deleted.")
            print("  tracks_removed: 0")
            print("  labels_removed: 0")
            return

        conn.execute("BEGIN")
        conn.execute(
            f"""
            DELETE FROM transition_labels
            WHERE track_a_file_path IN ({placeholders})
               OR track_b_file_path IN ({placeholders})
            """,
            tuple(candidates + candidates),
        )
        labels_removed = conn.total_changes

        conn.execute(
            f"DELETE FROM tracks WHERE file_path IN ({placeholders})",
            tuple(candidates),
        )
        tracks_removed = conn.total_changes - labels_removed

        conn.commit()

        print("[cleanup_tracks] DONE")
        print(f"  tracks_matched: {len(candidates)}")
        print(f"  tracks_removed: {tracks_removed}")
        print(f"  labels_removed: {labels_removed}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()

