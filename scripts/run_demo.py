from __future__ import annotations

from dataclasses import replace

from app.config import Settings, get_settings
from app.memory.service import MemoryEngine


def build_demo_settings() -> Settings:
    return replace(get_settings(), database_path="data/demo_memory.db")


def run_demo() -> None:
    settings = build_demo_settings()
    engine = MemoryEngine(settings)
    session_id = "demo_1000_turns"

    turns = [
        "My preferred language is Kannada.",
        "I can only take calls after 11 AM.",
    ]
    for turn in range(3, 999):
        turns.append(f"This is filler conversation turn {turn} on mixed topics.")
    turns.append("Can you call me tomorrow?")
    turns.append("Can we continue in my language?")

    inspect_turns = {1, 2, 999, 1000}

    try:
        for index, text in enumerate(turns, start=1):
            if index in inspect_turns:
                engine.drain_background(timeout_seconds=2.0)

            result = engine.process_user_turn(
                session_id=session_id,
                user_message=text,
                expose_memory=index in inspect_turns,
            )

            if index in inspect_turns:
                print(f"Turn {index} user: {text}")
                print(f"Turn {index} assistant: {result['response_text']}")
                print("Active memories:")
                for memory in result["active_memories"]:
                    origin_turn = memory.get("origin_turn", memory.get("source_turn"))
                    print(
                        f"  - {memory['memory_id']} | {memory.get('content', '')} "
                        f"(origin_turn={origin_turn}, score={memory.get('score', 0.0):.3f})"
                    )
                print()

        print("Demo complete: 1,000-turn conversation processed.")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    run_demo()
