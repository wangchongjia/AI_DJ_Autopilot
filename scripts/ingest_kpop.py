from __future__ import annotations

try:
    # Optional: load .env for local convenience (still works without it).
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from ai_dj_copilot.ingest import main


if __name__ == "__main__":
    main()

