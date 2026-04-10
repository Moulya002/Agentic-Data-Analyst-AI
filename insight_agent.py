from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from planner import AgentStep
from tools import ToolResult, quality_summary_text
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
        lines = [
            f"**Your question:** {user_query}",
            "",
            "**What we ran and found:**",
        ]
        for idx, output in enumerate(outputs, start=1):
            if idx <= len(steps):
                step = steps[idx - 1]
                label = f"[{step.tool}]" if mode == "technical" else f"Step {idx}"
                extra = f" | params={step.params}" if mode == "technical" else ""
                lines.append(f"- {label}{extra} {output.text}")
            else:
                lines.append(f"- (extra) {output.text}")
        lines.append("")
        lines.append(
            "**Takeaway:** Use the facts above to answer your question; ask for SQL, a specific column, or a chart if you need more detail."
        )
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
                "You are an insight agent. The user asked a specific question in the 'query' field. "
                "Answer THAT question directly using the tool results (numbers, tables, summaries). "
                "If the tools only partially answer the question, say what is missing and suggest one next step. "
                "If mode is simple, use plain language; if technical, include metrics, column names, and tool names. "
                "Do not give a generic dataset essay — tie sentences to the user's question."
            ),
            user_payload=payload,
            temperature=0.3,
        )

    def pipeline_insights(
        self,
        *,
        quality_report: Dict[str, Any],
        clean_summary: str,
        analysis_summary: str,
        chart_descriptions: List[str],
        mode: str = "simple",
    ) -> List[str]:
        """
        4–6 bullets after CSV upload pipeline. Uses Groq when enabled, else rules.
        """
        fallback = self._pipeline_insights_fallback(
            quality_report, clean_summary, analysis_summary, chart_descriptions, mode
        )
        if not self.llm.enabled:
            return fallback[:6]
        try:
            payload = {
                "quality_summary": quality_summary_text(quality_report) if quality_report else "",
                "cleaning": clean_summary,
                "analysis": analysis_summary,
                "charts": chart_descriptions,
                "mode": mode,
                "task": (
                    "Return ONLY a JSON array of 4 to 6 strings. "
                    "Each string is one business insight (no numbering prefix). "
                    "If mode is technical, include metrics and column names where helpful."
                ),
            }
            raw = self.llm.chat_json(
                system_prompt="You output valid JSON only: an array of strings.",
                user_payload=payload,
                temperature=0.35,
            )
            parsed = self._parse_string_list(raw)
            if parsed and 4 <= len(parsed) <= 8:
                return parsed[:6]
            if parsed:
                return (parsed + fallback)[:6]
        except Exception:
            pass
        return fallback[:6]

    def _parse_string_list(self, raw: str) -> Optional[List[str]]:
        text = (raw or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
            if isinstance(data, list):
                out = [str(x).strip() for x in data if str(x).strip()]
                return out or None
        except json.JSONDecodeError:
            start, end = text.find("["), text.rfind("]")
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start : end + 1])
                    if isinstance(data, list):
                        out = [str(x).strip() for x in data if str(x).strip()]
                        return out or None
                except json.JSONDecodeError:
                    pass
        lines: List[str] = []
        for line in text.splitlines():
            s = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip()
            if len(s) > 20:
                lines.append(s)
        return lines or None

    def _pipeline_insights_fallback(
        self,
        quality_report: Dict[str, Any],
        clean_summary: str,
        analysis_summary: str,
        chart_descriptions: List[str],
        mode: str,
    ) -> List[str]:
        bullets: List[str] = []
        if quality_report:
            bullets.append(quality_summary_text(quality_report))
        if clean_summary:
            first = clean_summary.strip().split("\n")[0]
            bullets.append(f"Cleaning applied: {first}")
        if analysis_summary:
            for line in analysis_summary.split("\n"):
                line = line.strip()
                if line.startswith("- ") and len(line) > 25:
                    bullets.append(line[2:].strip())
                elif "correlation" in line.lower() and len(line) > 25:
                    bullets.append(line)
        if chart_descriptions:
            bullets.append(
                f"Auto-visualization: {len(chart_descriptions)} chart(s) — "
                + "; ".join(chart_descriptions[:3])
            )
        bullets.append(
            "Next: use chat to ask for breakdowns by segment, time windows, or specific KPIs."
        )
        if mode == "technical":
            bullets.append(
                "Methods: duplicate detection via pandas; missing imputation uses median (numeric) "
                "and literal 'Unknown' (non-numeric); outliers flagged with 1.5×IQR on numeric columns."
            )
        seen = set()
        unique: List[str] = []
        for b in bullets:
            if b not in seen and len(b) > 15:
                seen.add(b)
                unique.append(b)
        while len(unique) < 4:
            unique.append("Explore correlations and segment-level cuts in chat for deeper validation.")
        return unique[:8]

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
            "task": (
                "Refine the answer so it clearly and directly addresses the user's query using the tool results. "
                "Remove generic filler. Return the refined answer only (no 'Query:' prefix)."
            ),
        }
        return self.llm.chat_json(
            system_prompt=(
                "You are a reflection agent. The answer must be specific to the user's question and grounded in the provided results."
            ),
            user_payload=payload,
            temperature=0.3,
        )
