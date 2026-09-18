import hashlib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "history.db")

_YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/|v/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)


def get_video_id(source: str) -> str:
    """Stable id for a video source: YouTube video id, or a hash for local files
    / unrecognized URL shapes."""
    match = _YOUTUBE_ID_RE.search(source)
    if match:
        return match.group(1)

    if source.startswith("http://") or source.startswith("https://"):
        return hashlib.sha256(source.encode()).hexdigest()[:16]

    try:
        stat = os.stat(source)
        key = f"{os.path.abspath(source)}|{stat.st_mtime}|{stat.st_size}"
    except OSError:
        key = source
    return hashlib.sha256(key.encode()).hexdigest()[:16]


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                source TEXT,
                title TEXT,
                processed_at TEXT,
                transcript TEXT,
                summary TEXT,
                action_items TEXT,
                key_decisions TEXT,
                open_questions TEXT,
                chunk_count INTEGER,
                chunks_json TEXT,
                language TEXT
            )
            """
        )


def save_video(
    video_id: str,
    source: str,
    title: str,
    transcript: str,
    summary: str,
    action_items: str,
    key_decisions: str,
    open_questions: str,
    chunk_count: int,
    chunks: list,
    language: str,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO videos
                (video_id, source, title, processed_at, transcript, summary,
                 action_items, key_decisions, open_questions, chunk_count, chunks_json, language)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_id,
                source,
                title,
                datetime.utcnow().isoformat(),
                transcript,
                summary,
                action_items,
                key_decisions,
                open_questions,
                chunk_count,
                json.dumps(chunks),
                language,
            ),
        )


def list_videos() -> list:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT video_id, title, processed_at, source FROM videos ORDER BY processed_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def load_video(video_id: str):
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["chunks"] = json.loads(result.pop("chunks_json") or "[]")
        return result
