from __future__ import annotations

import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .types import MemoryCandidate, MemoryStatus


class MemoryStore:
    def __init__(self, database_path: str) -> None:
        db_path = Path(database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(database_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._fts_enabled = False
        self._init_db()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(session_id, turn_id)
                );

                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    normalized_value TEXT NOT NULL,
                    subject TEXT,
                    relation TEXT,
                    object TEXT,
                    source_turn INTEGER NOT NULL,
                    last_used_turn INTEGER,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    supersedes_memory_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session_turn ON turns(session_id, turn_id);
                CREATE INDEX IF NOT EXISTS idx_memories_session_status ON memories(session_id, status);
                CREATE INDEX IF NOT EXISTS idx_memories_session_key_status ON memories(session_id, key, status);
                CREATE INDEX IF NOT EXISTS idx_embeddings_session ON memory_embeddings(session_id);
                """
            )

            try:
                self._conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                        memory_id UNINDEXED,
                        session_id UNINDEXED,
                        content
                    );
                    """
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False

            self._conn.commit()

    def next_turn_id(self, session_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(turn_id), 0) + 1 AS next_turn FROM turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return int(row["next_turn"])

    def add_turn(self, session_id: str, turn_id: int, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO turns(session_id, turn_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, turn_id, role, content, self._now()),
            )
            self._conn.commit()

    def get_recent_turns(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT turn_id, role, content
                FROM turns
                WHERE session_id = ?
                ORDER BY turn_id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        ordered = list(reversed(rows))
        return [dict(row) for row in ordered]

    @staticmethod
    def _fts_content(key: str, value: str, subject: str | None, relation: str | None, object_: str | None) -> str:
        parts = [key, value]
        if subject:
            parts.append(subject)
        if relation:
            parts.append(relation)
        if object_:
            parts.append(object_)
        return " ".join(parts)

    def _upsert_fts(self, memory_id: str, session_id: str, content: str) -> None:
        if not self._fts_enabled:
            return
        self._conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
        self._conn.execute(
            "INSERT INTO memory_fts(memory_id, session_id, content) VALUES (?, ?, ?)",
            (memory_id, session_id, content),
        )

    def upsert_memory(
        self,
        session_id: str,
        source_turn: int,
        candidate: MemoryCandidate,
        vector_payload: str,
    ) -> str:
        with self._lock:
            now = self._now()
            active_rows = self._conn.execute(
                """
                SELECT *
                FROM memories
                WHERE session_id = ? AND key = ? AND status = ?
                ORDER BY source_turn DESC
                """,
                (session_id, candidate.key, MemoryStatus.ACTIVE.value),
            ).fetchall()

            same_value_row = None
            for row in active_rows:
                if row["normalized_value"] == candidate.normalized_value:
                    same_value_row = row
                    break

            if same_value_row is not None:
                memory_id = same_value_row["memory_id"]
                for row in active_rows:
                    if row["memory_id"] != memory_id:
                        self._conn.execute(
                            "UPDATE memories SET status = ?, updated_at = ? WHERE memory_id = ?",
                            (MemoryStatus.DEPRECATED.value, now, row["memory_id"]),
                        )

                self._conn.execute(
                    """
                    UPDATE memories
                    SET confidence = ?, updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (max(float(same_value_row["confidence"]), candidate.confidence), now, memory_id),
                )
            else:
                supersedes_memory_id = active_rows[0]["memory_id"] if active_rows else None
                for row in active_rows:
                    self._conn.execute(
                        "UPDATE memories SET status = ?, updated_at = ? WHERE memory_id = ?",
                        (MemoryStatus.DEPRECATED.value, now, row["memory_id"]),
                    )

                memory_id = f"mem_{uuid4().hex[:12]}"
                self._conn.execute(
                    """
                    INSERT INTO memories(
                        memory_id,
                        session_id,
                        type,
                        key,
                        value,
                        normalized_value,
                        subject,
                        relation,
                        object,
                        source_turn,
                        last_used_turn,
                        confidence,
                        status,
                        supersedes_memory_id,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        session_id,
                        candidate.memory_type.value,
                        candidate.key,
                        candidate.value,
                        candidate.normalized_value,
                        candidate.subject,
                        candidate.relation,
                        candidate.object,
                        source_turn,
                        None,
                        candidate.confidence,
                        MemoryStatus.ACTIVE.value,
                        supersedes_memory_id,
                        now,
                        now,
                    ),
                )

            self._conn.execute(
                """
                INSERT INTO memory_embeddings(memory_id, session_id, vector, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    vector = excluded.vector,
                    updated_at = excluded.updated_at
                """,
                (memory_id, session_id, vector_payload, now),
            )

            self._upsert_fts(
                memory_id,
                session_id,
                self._fts_content(candidate.key, candidate.value, candidate.subject, candidate.relation, candidate.object),
            )
            self._conn.commit()
            return memory_id

    def fetch_active_memories_with_vectors(self, session_id: str) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                """
                SELECT m.*, e.vector
                FROM memories m
                LEFT JOIN memory_embeddings e ON e.memory_id = m.memory_id
                WHERE m.session_id = ? AND m.status = ?
                """,
                (session_id, MemoryStatus.ACTIVE.value),
            ).fetchall()

    def lexical_search(self, session_id: str, query: str, limit: int) -> list[sqlite3.Row]:
        terms = re.findall(r"[a-zA-Z0-9_]+", query.lower())
        if not terms:
            return []

        with self._lock:
            if self._fts_enabled:
                match_expression = " OR ".join(terms[:12])
                try:
                    rows = self._conn.execute(
                        """
                        SELECT m.*, bm25(memory_fts) AS lexical_score
                        FROM memory_fts
                        JOIN memories m ON m.memory_id = memory_fts.memory_id
                        WHERE memory_fts MATCH ?
                          AND memory_fts.session_id = ?
                          AND m.status = ?
                        ORDER BY lexical_score
                        LIMIT ?
                        """,
                        (match_expression, session_id, MemoryStatus.ACTIVE.value, limit),
                    ).fetchall()
                    return rows
                except sqlite3.OperationalError:
                    pass

            like_query = f"%{terms[0]}%"
            return self._conn.execute(
                """
                SELECT *, 1.0 AS lexical_score
                FROM memories
                WHERE session_id = ?
                  AND status = ?
                  AND (LOWER(key) LIKE ? OR LOWER(value) LIKE ?)
                LIMIT ?
                """,
                (session_id, MemoryStatus.ACTIVE.value, like_query, like_query, limit),
            ).fetchall()

    def mark_memories_used(self, memory_ids: list[str], turn_id: int) -> None:
        if not memory_ids:
            return

        placeholders = ",".join("?" for _ in memory_ids)
        params: list[Any] = [turn_id, self._now(), *memory_ids]

        with self._lock:
            self._conn.execute(
                f"UPDATE memories SET last_used_turn = ?, updated_at = ? WHERE memory_id IN ({placeholders})",
                tuple(params),
            )
            self._conn.commit()

    def list_memories(self, session_id: str, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    """
                    SELECT *
                    FROM memories
                    WHERE session_id = ? AND status = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (session_id, status, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT *
                    FROM memories
                    WHERE session_id = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()

        return [dict(row) for row in rows]

    def conflict_rate(self, session_id: str) -> float:
        with self._lock:
            total_keys_row = self._conn.execute(
                """
                SELECT COUNT(DISTINCT key) AS total_keys
                FROM memories
                WHERE session_id = ? AND status = ?
                """,
                (session_id, MemoryStatus.ACTIVE.value),
            ).fetchone()

            conflicting_keys_row = self._conn.execute(
                """
                SELECT COUNT(*) AS conflicting_keys
                FROM (
                    SELECT key
                    FROM memories
                    WHERE session_id = ? AND status = ?
                    GROUP BY key
                    HAVING COUNT(*) > 1
                )
                """,
                (session_id, MemoryStatus.ACTIVE.value),
            ).fetchone()

        total_keys = int(total_keys_row["total_keys"])
        conflicting_keys = int(conflicting_keys_row["conflicting_keys"])
        if total_keys == 0:
            return 0.0
        return conflicting_keys / total_keys

    def close(self) -> None:
        with self._lock:
            self._conn.close()
