from __future__ import annotations

import re
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class ConversationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'active',
                    handoff_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, message_id);
                """
            )

    def ensure_session(self, session_id: str | None = None) -> str:
        resolved = session_id or uuid.uuid4().hex
        if not SESSION_ID_PATTERN.fullmatch(resolved):
            raise ValueError("Invalid session ID.")
        timestamp = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions(session_id, created_at, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (resolved, timestamp, timestamp),
            )
        return resolved

    def append_exchange(self, session_id: str, user_content: str, assistant_content: str) -> None:
        timestamp = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, "user", user_content, timestamp),
            )
            connection.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, "assistant", assistant_content, timestamp),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (timestamp, session_id),
            )

    def append_message(self, session_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Conversation role must be user or assistant.")
        timestamp = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, timestamp),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (timestamp, session_id),
            )

    def history(self, session_id: str, limit: int = 20) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content FROM (
                    SELECT message_id, role, content
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY message_id DESC
                    LIMIT ?
                )
                ORDER BY message_id
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def request_handoff(self, session_id: str, reason: str) -> dict[str, Any]:
        timestamp = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions
                SET status = 'waiting_human', handoff_reason = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (reason, timestamp, session_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(session_id)
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT session_id, status, handoff_reason, created_at, updated_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def transcript(self, session_id: str) -> dict[str, Any] | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        session["messages"] = self.history(session_id, limit=100)
        return session
