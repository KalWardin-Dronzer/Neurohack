"""The memory service as an MCP server, so any MCP client (Claude Code, Claude Desktop, Cursor) can use the store.

    python -m app.mcp_server                       stdio, for a client that launches the server itself
    python -m app.mcp_server --http --port 8765    streamable HTTP at http://127.0.0.1:8765/mcp

Settings come from .env, as for the HTTP service, and the two can share one database file.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from app.config import get_settings
from app.memory.context import render, serialize_memories, serialize_turns
from app.memory.service import MemoryEngine

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTIONS = (
    "Long-term memory for conversations. Call `remember` with what the user says that is worth keeping, and "
    "`recall` before answering anything that may depend on earlier conversations. `recall` returns memories "
    "(typed facts with dates) and the earlier turns most similar to the query. Use one session_id per user."
)


def _memory_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_id": record["memory_id"],
        "type": record["type"],
        "subject": record["subject"],
        "content": record["value"],
        "event_date": record["event_date"],
        "evidence": record["evidence"],
        "status": record["status"],
        "source_turn": record["source_turn"],
    }


def build_server(engine: MemoryEngine, host: str = "127.0.0.1", port: int = 8765) -> FastMCP:
    server = FastMCP("neurohack-memory", instructions=INSTRUCTIONS, host=host, port=port)

    async def run(method: str, *args, **kwargs):
        # Extraction can call an LLM for seconds; a worker thread keeps the server answering meanwhile.
        return await anyio.to_thread.run_sync(lambda: getattr(engine, method)(*args, **kwargs))

    @server.tool(annotations=ToolAnnotations(title="Remember", readOnlyHint=False, destructiveHint=False))
    async def remember(text: str, session_id: str = "default", speaker: str = "user") -> dict[str, Any]:
        """Store one conversation turn and extract the memories in it now: facts, preferences, plans and dated
        events, each citing the turn. Returns the turn id and the memories stored or updated."""
        result = await run("remember", session_id, text, speaker=speaker)
        return {"turn_id": result["turn_id"], "memories": [_memory_record(r) for r in result["memories"]]}

    @server.tool(annotations=ToolAnnotations(title="Recall", readOnlyHint=True))
    async def recall(query: str, session_id: str = "default", memories: int = 10, turns: int = 5) -> dict[str, Any]:
        """Retrieve what is remembered about a query: the most relevant memories (with dates) and the earlier raw
        turns most similar to it. `context` renders both as one block to put in a prompt. The default of 10
        memories and 5 turns is the configuration measured on the LoCoMo benchmark."""
        recalled = await run("recall", session_id, query, memories=max(0, memories), turns=max(0, turns))
        return {
            "context": render(recalled),
            "memories": serialize_memories(recalled.memories),
            "turns": serialize_turns(recalled.turns),
        }

    @server.tool(annotations=ToolAnnotations(title="Forget", readOnlyHint=False, destructiveHint=True, idempotentHint=True))
    async def forget(memory_id: str, session_id: str = "default") -> dict[str, Any]:
        """Delete one memory by id and redact the turns it was extracted from, so recall cannot return it again.
        Other memories drawn from those turns are kept."""
        forgotten = await run("forget", session_id, memory_id)
        if forgotten is None:
            raise ValueError(f"session {session_id!r} has no memory {memory_id!r}")
        return {"forgotten": _memory_record(forgotten["memory"]), "redacted_turns": forgotten["redacted_turns"]}

    @server.tool(annotations=ToolAnnotations(title="List memories", readOnlyHint=True))
    async def list_memories(session_id: str = "default", limit: int = 50) -> dict[str, Any]:
        """List a session's active memories, most recently updated first."""
        records = await run("list_memories", session_id, status="active", limit=max(1, min(limit, 1000)))
        return {"memories": [_memory_record(r) for r in records]}

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Neurohack memory as an MCP server")
    parser.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    # Under stdio, stdout carries the protocol, so logs must go to stderr.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    # Clients launch the server from any directory; relative paths in .env, such as MEMORY_DB_PATH, mean the
    # repository root, as they do for the HTTP service.
    os.chdir(ROOT)

    # The engine loads before the server starts reading stdin, which delays the handshake by the embedding model's
    # load time. Loading it in a background thread instead hangs on Windows: once a thread is blocked reading the
    # stdin pipe, a native library that sentence-transformers loads (scipy, through scikit-learn) never finishes
    # loading.
    engine = MemoryEngine(get_settings())
    try:
        build_server(engine, args.host, args.port).run("streamable-http" if args.http else "stdio")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
