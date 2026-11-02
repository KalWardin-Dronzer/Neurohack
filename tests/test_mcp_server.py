"""The MCP server end to end: launched over stdio as a client would launch it, then driven through the protocol."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def _session_flow(db_path: Path) -> dict:
    # Every setting the server reads is pinned, so a developer's .env cannot switch it to an LLM or a real database.
    env = {
        **os.environ,
        "MEMORY_DB_PATH": str(db_path),
        "MEMORY_EXTRACTOR": "regex",
        "ENABLE_LLM_RESPONSE": "false",
        "EMBEDDING_BACKEND": "hashed",
        "LLM_CACHE_PATH": "",
        "RETRIEVAL_TOP_K": "10",
        "TURN_TOP_K": "5",
    }
    # Launched from another directory, as a client launches it, with the repository on PYTHONPATH as the README says.
    env["PYTHONPATH"] = str(ROOT)
    params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"], env=env, cwd=str(db_path.parent))
    out: dict = {}

    async def call(session: ClientSession, tool: str, **arguments):
        result = await session.call_tool(tool, arguments)
        payload = result.structuredContent or json.loads(result.content[0].text)
        return result.isError, payload

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            out["tools"] = {tool.name: tool for tool in tools}
            out["remember"] = await call(session, "remember", text="My preferred language is Kannada.", session_id="u1")
            await call(session, "remember", text="I live in Pune.", session_id="u1")
            out["recall"] = await call(session, "recall", query="What is my preferred language?", session_id="u1")
            out["other_user"] = await call(session, "recall", query="What is my preferred language?", session_id="u2")
            memory_id = out["remember"][1]["memories"][0]["memory_id"]
            out["forget"] = await call(session, "forget", memory_id=memory_id, session_id="u1")
            out["after"] = await call(session, "recall", query="What is my preferred language?", session_id="u1")
            out["list"] = await call(session, "list_memories", session_id="u1")
            result = await session.call_tool("forget", {"memory_id": memory_id, "session_id": "u1"})
            out["forget_again"] = (result.isError, result.content[0].text)
    return out


def test_mcp_server_over_stdio(tmp_path) -> None:
    out = anyio.run(_session_flow, tmp_path / "mcp.db")

    tools = out["tools"]
    assert set(tools) == {"remember", "recall", "forget", "list_memories"}
    assert tools["recall"].annotations.readOnlyHint and tools["forget"].annotations.destructiveHint

    error, remembered = out["remember"]
    assert not error and remembered["turn_id"] == 1
    assert remembered["memories"][0]["content"] == "Kannada" and remembered["memories"][0]["evidence"] == ["T1"]

    error, recalled = out["recall"]
    assert not error
    assert recalled["memories"][0]["content"] == "User prefers Kannada"
    assert recalled["turns"][0]["content"] == "My preferred language is Kannada."
    assert recalled["context"].startswith("Memories:\n- User prefers Kannada")
    assert out["other_user"][1] == {"context": "", "memories": [], "turns": []}

    error, forgotten = out["forget"]
    assert not error and forgotten["redacted_turns"] == [1]
    error, after = out["after"]
    assert "Kannada" not in after["context"] and [t["turn_id"] for t in after["turns"]] == [2]
    assert [m["content"] for m in out["list"][1]["memories"]] == ["Pune"]
    assert out["forget_again"][0] and "has no memory" in out["forget_again"][1]
