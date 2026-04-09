from __future__ import annotations

from typing import List

from planner import AgentStep
from tools import ToolResult
from utils.llm_client import GroqLLM


class InsightAgent:
    """
    Insight Agent:
    - converts execution results into simple human-readable insights
    """

    def __init__(self):
        self.llm = GroqLLM()

    def summarize(self, user_query: str, steps: List[AgentStep], outputs: List[ToolResult], mode: str = "simple") -> str:
        if not self.llm.enabled:
            return self._fallback_summary(user_query, steps, outputs, mode=mode)
        try:
            return self._llm_summary(user_query, steps, outputs, mode=mode)
        except Exception:
            return self._fallback_summary(user_query, steps, outputs, mode=mode)

    def _fallback_summary(self, user_query: str, steps: List[AgentStep], outputs: List[ToolResult], mode: str = "simple") -> str:
        lines = [f"Query: {user_query}", "", "Insights:"]
        for idx, (step, output) in enumerate(zip(steps, outputs), start=1):
            if mode == "technical":
                lines.append(f"{idx}. [{step.tool}] {output.text} | params={step.params}")
            else:
                lines.append(f"{idx}. {output.text}")
        lines.append("")
        lines.append("Next step: ask for deeper SQL cuts or targeted charts for specific columns.")
        return "\n".join(lines)

    def _llm_summary(self, user_query: str, steps: List[AgentStep], outputs: List[ToolResult], mode: str = "simple") -> str:
        payload = {
            "query": user_query,
            "plan": [{"tool": s.tool, "params": s.params, "reason": s.reason} for s in steps],
            "results": [o.text for o in outputs],
            "mode": mode,
        }
        return self.llm.chat_json(
            system_prompt=(
                "You are an insight agent. Explain results in business language. "
                "If mode is simple, keep beginner-friendly. If technical, include metrics and tool rationale. "
                "Keep concise and actionable."
            ),
            user_payload=payload,
            temperature=0.3,
        )

    def reflect_and_refine(
        self,
        user_query: str,
        initial_answer: str,
        steps: List[AgentStep],
        outputs: List[ToolResult],
        mode: str = "simple",
    ) -> str:
        if not self.llm.enabled:
            return initial_answer
        try:
            return self._llm_reflection(user_query, initial_answer, steps, outputs, mode)
        except Exception:
            return initial_answer

    def _llm_reflection(
        self,
        user_query: str,
        initial_answer: str,
        steps: List[AgentStep],
        outputs: List[ToolResult],
        mode: str,
    ) -> str:
        payload = {
            "query": user_query,
            "initial_answer": initial_answer,
            "plan": [{"tool": s.tool, "reason": s.reason, "params": s.params} for s in steps],
            "results": [o.text for o in outputs],
            "mode": mode,
            "task": "Review the answer, fix weak claims, improve clarity and recommendations, then return refined answer only.",
        }
        return self.llm.chat_json(
            system_prompt="You are a reflection agent that improves analytical responses.",
            user_payload=payload,
            temperature=0.3,
        )
