from __future__ import annotations

import logging

from app.llm.client import LLMClient, LLMError

from .context import render
from .types import MemoryMatch, Recall, TurnMatch

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant with long-term memory of this user. Use the memories and earlier turns below "
    "implicitly; do not list them unless asked. When a newer memory contradicts an older one, trust the newer one."
)


class ResponseGenerator:
    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client

    def generate(
        self,
        user_message: str,
        recent_turns: list[dict[str, str]],
        memories: list[MemoryMatch],
        earlier_turns: list[TurnMatch] | None = None,
    ) -> str:
        if self._client is not None:
            recall = Recall(memories=memories, turns=earlier_turns or [])
            response = self._generate_with_llm(user_message=user_message, recent_turns=recent_turns, recall=recall)
            if response:
                return response

        return self._generate_fallback(user_message=user_message, memories=memories)

    def _generate_with_llm(
        self,
        user_message: str,
        recent_turns: list[dict[str, str]],
        recall: Recall,
    ) -> str | None:
        memory_context = render(recall) or "Memories:\n- none"
        messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{memory_context}"}]
        for turn in recent_turns[-8:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        # The store already holds the current turn, so it usually closes recent_turns; v1 sent it twice.
        if not recent_turns or recent_turns[-1]["content"] != user_message:
            messages.append({"role": "user", "content": user_message})

        try:
            return self._client.chat(messages, temperature=0.2, max_tokens=300).text
        except LLMError as exc:
            logger.warning("LLM response failed (%s); using the rule-based reply", exc)
            return None

    def _generate_fallback(self, user_message: str, memories: list[MemoryMatch]) -> str:
        user_lower = user_message.lower().strip()
        memory_by_key = {memory.key: memory.value for memory in memories}

        if self._is_greeting(user_lower):
            return "Hi! How can I help you today?"

        if self._is_identity_question(user_lower):
            return "I'm your long-form memory assistant."

        if self._is_user_name_question(user_lower):
            user_name = memory_by_key.get("user_name")
            if user_name:
                return f"Your name is {user_name}."
            return "I don't have your name in memory yet."

        if "call me" in user_lower:
            call_pref = memory_by_key.get("call_time_preference")
            if call_pref:
                return f"Sure. I can call you {call_pref}."
            return "Sure. I can do that. What time works best for you?"

        if "preferred language" in user_lower or "my language" in user_lower or "speak" in user_lower:
            preferred_language = memory_by_key.get("preferred_language")
            if preferred_language:
                return f"We can continue in {preferred_language}."
            return "I can continue in your preferred language once you tell me which one."

        if self._looks_like_memory_statement(user_lower):
            return "Understood. I noted that and will use it in future responses."

        return "I can help with reminders, preferences, and planning. What would you like to do?"

    @staticmethod
    def _is_greeting(text: str) -> bool:
        return text in {"hi", "hello", "hey", "yo", "good morning", "good afternoon", "good evening"}

    @staticmethod
    def _is_identity_question(text: str) -> bool:
        return text in {"what is your name?", "what's your name?", "who are you?"}

    @staticmethod
    def _is_user_name_question(text: str) -> bool:
        return text in {"what is my name?", "what's my name?"}

    @staticmethod
    def _looks_like_memory_statement(text: str) -> bool:
        return any(
            signal in text
            for signal in (
                "my preferred",
                "my favorite",
                "my name is",
                "i live in",
                "i can only",
                "remember to",
                "don't let me forget",
                "i will",
            )
        )
