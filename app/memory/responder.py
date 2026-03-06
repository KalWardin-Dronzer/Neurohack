from __future__ import annotations

import os

from .types import MemoryMatch


class ResponseGenerator:
    def __init__(self, model: str, enable_openai: bool = False) -> None:
        self.model = model
        self.enable_openai = enable_openai
        self._client = self._create_openai_client() if enable_openai else None

    @staticmethod
    def _create_openai_client():
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI  # type: ignore

            return OpenAI(api_key=api_key)
        except Exception:
            return None

    def generate(
        self,
        user_message: str,
        recent_turns: list[dict[str, str]],
        memories: list[MemoryMatch],
    ) -> str:
        if self._client is not None:
            response = self._generate_with_openai(user_message=user_message, recent_turns=recent_turns, memories=memories)
            if response:
                return response

        return self._generate_fallback(user_message=user_message, memories=memories)

    def _generate_with_openai(
        self,
        user_message: str,
        recent_turns: list[dict[str, str]],
        memories: list[MemoryMatch],
    ) -> str | None:
        memory_context = "\n".join(
            f"- {memory.key}: {memory.value} (confidence={memory.confidence:.2f})" for memory in memories
        )
        if not memory_context:
            memory_context = "- none"

        system_prompt = (
            "You are a helpful assistant. Use provided memory context implicitly; "
            "do not list memories unless directly asked."
        )

        messages = [{"role": "system", "content": system_prompt}]
        for turn in recent_turns[-8:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "system", "content": f"Relevant long-term memory:\n{memory_context}"})
        messages.append({"role": "user", "content": user_message})

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=220,
            )
            return completion.choices[0].message.content
        except Exception:
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
