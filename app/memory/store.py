from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .types import MemoryCandidate, MemoryStatus, MemoryType

SCHEMA_VERSION = 2

_STOPWORDS = frozenset(
    """
    a an and are as at be been but by can did do does for from had has have he her hers him his how i if in
    into is it its me my of on or our she so than that the their them then there they this to was we were
    what when where which who whom why will with would you your
    """.split()
)


class MemoryStore:
    def __init__(self, database_path: str) -> None:
        db_path = Path(database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._path = database_path
        self._conn = sqlite3.connect(database_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._fts_enabled = False
        # Bumped on every write so the retriever knows when its cached vector matrix is stale.
        self._versions: dict[str, int] = {}
        self._init_db()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _init_db(self) -> None:
        with self._lock:
            version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            has_memories = self._conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'memories'"
            ).fetchone()[0]
            if has_memories and version < SCHEMA_VERSION:
                raise RuntimeError(
                    f"{self._path} was created by v1 (schema {version}); delete it or set MEMORY_DB_PATH to a new file"
                )

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
                    subject TEXT NOT NULL DEFAULT 'user',
                    relation TEXT,
                    object TEXT,
                    event_date TEXT,
                    evidence TEXT NOT NULL DEFAULT '[]',
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
                    model_id TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session_turn ON turns(session_id, turn_id);
                CREATE INDEX IF NOT EXISTS idx_memories_session_status ON memories(session_id, status);
                CREATE INDEX IF NOT EXISTS idx_memories_slot ON memories(session_id, subject, key, status);
                CREATE INDEX IF NOT EXISTS idx_embeddings_session ON memory_embeddings(session_id);
                """
            )

            try:
                self._conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                        memory_id UNINDEXED,
                        session_id UNINDEXED,
                        content,
                        tokenize = 'porter unicode61'
                    );
                    """
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False

            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()

    def _touch(self, session_id: str) -> None:
        self._versions[session_id] = self._versions.get(session_id, 0) + 1

    def memory_version(self, session_id: str) -> int:
        with self._lock:
            return self._versions.get(session_id, 0)

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
    def _fts_content(candidate: MemoryCandidate) -> str:
        parts = [candidate.key.replace("_", " "), candidate.value, candidate.subject]
        if candidate.relation:
            parts.append(candidate.relation)
        if candidate.object:
            parts.append(candidate.object)
        return " ".join(parts)

    def _upsert_fts(self, memory_id: str, session_id: str, content: str) -> None:
        if not self._fts_enabled:
            return
        self._conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
        self._conn.execute(
            "INSERT INTO memory_fts(memory_id, session_id, content) VALUES (?, ?, ?)",
            (memory_id, session_id, content),
        )

    def _put_embedding_locked(self, memory_id: str, session_id: str, vector: bytes, model_id: str, dim: int) -> None:
        self._conn.execute(
            """
            INSERT INTO memory_embeddings(memory_id, session_id, model_id, dim, vector, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                session_id = excluded.session_id,
                model_id = excluded.model_id,
                dim = excluded.dim,
                vector = excluded.vector,
                updated_at = excluded.updated_at
            """,
            (memory_id, session_id, model_id, dim, vector, self._now()),
        )

    def upsert_memory(
        self,
        session_id: str,
        source_turn: int,
        candidate: MemoryCandidate,
        vector: bytes,
        model_id: str,
        dim: int,
    ) -> str:
        with self._lock:
            now = self._now()
            if candidate.is_event:
                memory_id = self._upsert_event(session_id, source_turn, candidate, now)
            else:
                memory_id = self._upsert_slot(session_id, source_turn, candidate, now)

            self._put_embedding_locked(memory_id, session_id, vector, model_id, dim)
            self._upsert_fts(memory_id, session_id, self._fts_content(candidate))
            self._touch(session_id)
            self._conn.commit()
            return memory_id

    def _upsert_event(self, session_id: str, source_turn: int, candidate: MemoryCandidate, now: str) -> str:
        duplicate = self._conn.execute(
            """
            SELECT memory_id, evidence, confidence, event_date
            FROM memories
            WHERE session_id = ? AND subject = ? AND normalized_value = ? AND status = ?
            """,
            (session_id, candidate.subject, candidate.normalized_value, MemoryStatus.ACTIVE.value),
        ).fetchone()
        if duplicate is None:
            return self._insert(session_id, source_turn, candidate, supersedes_memory_id=None, now=now)

        evidence = sorted(set(json.loads(duplicate["evidence"])) | set(candidate.evidence))
        self._conn.execute(
            """
            UPDATE memories
            SET evidence = ?, confidence = ?, event_date = COALESCE(event_date, ?), updated_at = ?
            WHERE memory_id = ?
            """,
            (
                json.dumps(evidence),
                max(float(duplicate["confidence"]), candidate.confidence),
                candidate.event_date,
                now,
                duplicate["memory_id"],
            ),
        )
        return str(duplicate["memory_id"])

    def _upsert_slot(self, session_id: str, source_turn: int, candidate: MemoryCandidate, now: str) -> str:
        active_rows = self._conn.execute(
            """
            SELECT *
            FROM memories
            WHERE session_id = ? AND subject = ? AND key = ? AND status = ?
            ORDER BY source_turn DESC
            """,
            (session_id, candidate.subject, candidate.key, MemoryStatus.ACTIVE.value),
        ).fetchall()

        same_value_row = next(
            (row for row in active_rows if row["normalized_value"] == candidate.normalized_value), None
        )
        if same_value_row is not None:
            memory_id = same_value_row["memory_id"]
            for row in active_rows:
                if row["memory_id"] != memory_id:
                    self._deprecate(row["memory_id"], now)
            evidence = sorted(set(json.loads(same_value_row["evidence"])) | set(candidate.evidence))
            self._conn.execute(
                "UPDATE memories SET confidence = ?, evidence = ?, updated_at = ? WHERE memory_id = ?",
                (max(float(same_value_row["confidence"]), candidate.confidence), json.dumps(evidence), now, memory_id),
            )
            return str(memory_id)

        supersedes_memory_id = active_rows[0]["memory_id"] if active_rows else None
        for row in active_rows:
            self._deprecate(row["memory_id"], now)
        return self._insert(session_id, source_turn, candidate, supersedes_memory_id=supersedes_memory_id, now=now)

    def _deprecate(self, memory_id: str, now: str) -> None:
        self._conn.execute(
            "UPDATE memories SET status = ?, updated_at = ? WHERE memory_id = ?",
            (MemoryStatus.DEPRECATED.value, now, memory_id),
        )
        if self._fts_enabled:
            self._conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))

    def _insert(
        self,
        session_id: str,
        source_turn: int,
        candidate: MemoryCandidate,
        supersedes_memory_id: str | None,
        now: str,
    ) -> str:
        memory_id = f"mem_{uuid4().hex[:12]}"
        self._conn.execute(
            """
            INSERT INTO memories(
                memory_id, session_id, type, key, value, normalized_value, subject, relation, object,
                event_date, evidence, source_turn, last_used_turn, confidence, status,
                supersedes_memory_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                candidate.event_date,
                json.dumps(sorted(set(candidate.evidence))),
                source_turn,
                None,
                candidate.confidence,
                MemoryStatus.ACTIVE.value,
                supersedes_memory_id,
                now,
                now,
            ),
        )
        return memory_id

    def fetch_active_memories_with_vectors(self, session_id: str) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                """
                SELECT m.*, e.vector, e.model_id, e.dim
                FROM memories m
                LEFT JOIN memory_embeddings e ON e.memory_id = m.memory_id
                WHERE m.session_id = ? AND m.status = ?
                """,
                (session_id, MemoryStatus.ACTIVE.value),
            ).fetchall()

    def memories_needing_embedding(self, model_id: str) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                """
                SELECT m.memory_id, m.session_id, m.key, m.value
                FROM memories m
                LEFT JOIN memory_embeddings e ON e.memory_id = m.memory_id
                WHERE e.memory_id IS NULL OR e.model_id != ?
                """,
                (model_id,),
            ).fetchall()

    def put_embedding(self, memory_id: str, session_id: str, vector: bytes, model_id: str, dim: int) -> None:
        with self._lock:
            self._put_embedding_locked(memory_id, session_id, vector, model_id, dim)
            self._touch(session_id)
            self._conn.commit()

    @staticmethod
    def query_terms(query: str, limit: int = 24) -> list[str]:
        terms: list[str] = []
        for term in re.findall(r"[a-zA-Z0-9_]+", query.lower()):
            if term not in _STOPWORDS and term not in terms:
                terms.append(term)
        return terms[:limit]

    def lexical_search(self, session_id: str, query: str, limit: int) -> list[sqlite3.Row]:
        """Return active memories matching the query, best first.

        `lexical_score` is the negated FTS5 bm25() value, so higher means more relevant. bm25()
        itself is smaller (more negative) for better matches; v1 mapped it to 1/(1+|x|) and so
        ranked the strongest matches lowest.
        """
        terms = self.query_terms(query)
        if not terms:
            return []

        with self._lock:
            if self._fts_enabled:
                try:
                    return self._conn.execute(
                        """
                        SELECT m.*, -bm25(memory_fts) AS lexical_score
                        FROM memory_fts
                        JOIN memories m ON m.memory_id = memory_fts.memory_id
                        WHERE memory_fts MATCH ?
                          AND memory_fts.session_id = ?
                          AND m.status = ?
                        ORDER BY lexical_score DESC
                        LIMIT ?
                        """,
                        (" OR ".join(terms), session_id, MemoryStatus.ACTIVE.value, limit),
                    ).fetchall()
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

    def active_keys(self, session_id: str, limit: int = 150) -> list[str]:
        """Most recent slot keys as `subject.key`, so an extractor can reuse them for updates."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT subject, key, MAX(source_turn) AS last_turn
                FROM memories
                WHERE session_id = ? AND status = ? AND type != ?
                GROUP BY subject, key
                ORDER BY last_turn DESC
                LIMIT ?
                """,
                (session_id, MemoryStatus.ACTIVE.value, MemoryType.EVENT.value, limit),
            ).fetchall()
        return [f"{row['subject']}.{row['key']}" for row in rows]

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

        records = []
        for row in rows:
            record = dict(row)
            record["evidence"] = json.loads(record["evidence"])
            records.append(record)
        return records

    def conflict_rate(self, session_id: str) -> float:
        """Share of slot keys with more than one active value. Events are excluded: they accumulate."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS total_keys, COALESCE(SUM(n > 1), 0) AS conflicting_keys
                FROM (
                    SELECT COUNT(*) AS n
                    FROM memories
                    WHERE session_id = ? AND status = ? AND type != ?
                    GROUP BY subject, key
                )
                """,
                (session_id, MemoryStatus.ACTIVE.value, MemoryType.EVENT.value),
            ).fetchone()

        total_keys = int(row["total_keys"])
        if total_keys == 0:
            return 0.0
        return int(row["conflicting_keys"]) / total_keys

    def close(self) -> None:
        with self._lock:
            self._conn.close()
