from langchain_core.messages import SystemMessage


class ContextManager:
    def __init__(self, llm, keep_last: int = 5, budget: int = 32000):
        self._llm = llm
        self._budget = budget
        self._keep_last = keep_last

    @property
    def keep_last(self) -> int:
        return self._keep_last

    async def condense(self, messages: list, existing_summary: str = "") -> str | None:
        total = self._count_tokens(messages)
        summary_tokens = len(existing_summary) // 2
        if total + summary_tokens <= self._budget:
            return None

        if len(messages) <= self._keep_last:
            return None

        old = messages[: -self._keep_last]
        text = "\n".join(
            f"{m.type}: {m.content[:200]}"
            for m in old
            if hasattr(m, "content") and m.content
        )

        if not text.strip():
            return None

        if existing_summary:
            prompt = (
                f"Update this structured fact list with new information from the conversation.\n"
                f"Output ONLY bullet points starting with '- '.\n\n"
                f"Current summary:\n{existing_summary}\n\n"
                f"New conversation:\n{text}\n\n"
                f"Updated summary:"
            )
        else:
            prompt = (
                f"Extract key facts from this conversation as bullet points starting with '- '.\n"
                f"Focus on user details, topics discussed, preferences.\n"
                f"Do NOT add any headers or preamble:\n\n{text}"
            )

        summary = await self._llm.ainvoke([SystemMessage(content=prompt)])
        return summary.content

    def _count_tokens(self, messages: list) -> int:
        return sum(len(str(m.content or "")) // 2 for m in messages)
