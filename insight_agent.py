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
                "You are a senior data analyst speaking to a business stakeholder. "
                "Answer the user's question using the tool results (numbers, column names, risks). "
                "Write like narrative bullets, not system logs — never say phrases like "
                "'Dataset summary complete', 'SQL query executed', or 'report generated'. "
                "If mode is simple, plain English; if technical, add metrics and method hints. "
                "Stay anchored to the user's question; no generic filler essay."
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
                    "Each string sounds like a real analyst (e.g. 'Revenue concentrates in …', 'West region is softening …'). "
                    "No robotic status text. If mode is technical, add metrics and column names."
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

    def business_recommendations(
        self,
        insights: List[str],
        quality_report: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        3–5 actionable recommendations for portfolio / demo (LLM when available).
        """
        quality_report = quality_report or {}
        recs: List[str] = []

        rows = int(quality_report.get("rows", 0) or 0)
        mb = quality_report.get("missing_by_column") or {}
        if isinstance(mb, dict) and mb and rows > 0:
            worst = max(mb.items(), key=lambda x: int(x[1] or 0))
            col, cnt = worst[0], int(worst[1] or 0)
            pct = 100.0 * cnt / max(1, rows)
            if cnt > 0:
                recs.append(
                    f"Prioritize fixing or imputing **`{col}`** — it has the largest missing footprint "
                    f"({cnt:,} cells, ~{pct:.0f}% of rows touched) before trusting forecasts."
                )

        dups = int(quality_report.get("duplicate_rows", 0) or 0)
        if rows > 0 and dups > 0:
            dpct = 100.0 * dups / rows
            if dpct >= 0.5:
                recs.append(
                    f"Tighten ingestion or dedup rules — **{dups:,} duplicate rows** (~{dpct:.1f}%) can inflate totals and distort rankings."
                )

        anomalies = quality_report.get("anomalies") or []
        if anomalies:
            recs.append(
                "Validate extreme values flagged as outliers with the owning team before excluding them from executive KPIs."
            )

        llm_recs: List[str] = []
        if self.llm.enabled:
            try:
                raw = self.llm.chat_json(
                    system_prompt='Return ONLY valid JSON: an array of 3 to 5 short actionable business recommendations.',
                    user_payload={
                        "insights_so_far": insights[:8],
                        "quality_summary": quality_summary_text(quality_report) if quality_report else "",
                        "task": "Recommendations must be specific (segments, regions, data fixes), not vague.",
                    },
                    temperature=0.35,
                )
                llm_recs = self._parse_string_list(raw) or []
            except Exception:
                llm_recs = []

        templates = [
            "Double down on the **top-performing segment** from the bar chart and replicate its playbook in weaker areas.",
            "Set up a **simple monthly tracker** on the time-series metric so you catch slowdowns before quarter-end.",
            "Align **sales and finance** on the worst missing-value columns so margin models stay defensible.",
        ]

        if len(llm_recs) >= 3:
            merged = llm_recs[:5]
        else:
            merged = (llm_recs + recs + templates)[:8]

        seen: set[str] = set()
        out: List[str] = []
        for r in merged:
            if r and r not in seen:
                seen.add(r)
                out.append(r)
        return out[:5]

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
