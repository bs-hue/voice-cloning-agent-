"""SQLite persistence layer for SwaraLab local voice cloning studio."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ["SWARALAB_DATA_DIR"]) if os.environ.get("SWARALAB_DATA_DIR") else (ROOT / "data")
DB_PATH = Path(os.environ["SWARALAB_DB_PATH"]) if os.environ.get("SWARALAB_DB_PATH") else (DATA_DIR / "studio.db")
SCHEMA_PATH = ROOT / "replica" / "schema.sql"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        conn.executescript(schema_sql)

        # Ensure supported inference parameter columns exist on generation_jobs
        existing_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(generation_jobs)").fetchall()
        }
        migrations = [
            ("exaggeration", "ALTER TABLE generation_jobs ADD COLUMN exaggeration real NOT NULL DEFAULT 0.5"),
            ("cfg_weight", "ALTER TABLE generation_jobs ADD COLUMN cfg_weight real NOT NULL DEFAULT 0.5"),
            ("temperature", "ALTER TABLE generation_jobs ADD COLUMN temperature real NOT NULL DEFAULT 0.8"),
            ("precision_used", "ALTER TABLE generation_jobs ADD COLUMN precision_used text NOT NULL DEFAULT 'float32'"),
        ]
        for col, ddl in migrations:
            if col not in existing_cols:
                conn.execute(ddl)

        # Default app_settings row uses cpu (float32) and 320-char chunks for natural multi-sentence prosody
        now = utc_now_iso()
        conn.execute(
            """
            INSERT OR IGNORE INTO app_settings (id, preferred_device, chunk_max_chars, updated_at)
            VALUES (1, 'cpu', 320, ?)
            """,
            (now,),
        )
        conn.execute(
            "UPDATE app_settings SET chunk_max_chars = 320, updated_at = ? WHERE id = 1 AND chunk_max_chars = 180",
            (now,),
        )
        conn.commit()
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None
