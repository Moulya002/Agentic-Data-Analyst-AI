from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from groq import APIStatusError, AuthenticationError

from utils.llm_client import GroqLLM
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentStep:
    tool: str
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class PlannerAgent:
    """
    Planner Agent:
    - accepts user query
    - creates executable step-by-step plan
    """

    def __init__(self):
        self.llm = GroqLLM()

    def plan(self, query: str, columns: List[str], memory_context: List[Dict[str, str]]) -> List[AgentStep]:
        if not self.llm.enabled:
            return self._fallback_plan(query, columns)

        system_prompt = (
            "You are a planning agent for a data analyst system. "
            "Return ONLY valid JSON list of steps with keys: tool, params, reason.\n"
            "Allowed tools: summarize_dataset, clean_data, run_sql, generate_plot, data_quality_report, generate_automatic_visualizations.\n"
            "If SQL is needed, table name is df.\n"
            "Keep steps minimal (1-4)."
        )
        user_prompt = {
            "query": query,
            "columns": columns,
            "memory": memory_context,
            "examples": [
                {"tool": "summarize_dataset", "params": {}, "reason": "Understand dataset first"},
                {"tool": "clean_data", "params": {}, "reason": "Fix obvious quality issues"},
                {"tool": "generate_plot", "params": {"x_col": "sales", "chart_type": "histogram"}, "reason": "Visualize distribution"},
            ],
        }
        try:
            raw = self.llm.chat_json(system_prompt=system_prompt, user_payload=user_prompt, temperature=0.3)
        except AuthenticationError as exc:
            logger.warning("Groq auth failed (planner); using fallback plan: %s", exc)
            return self._fallback_plan(query, columns)
        except APIStatusError as exc:
            logger.warning("Groq API error (planner); using fallback plan: %s", exc)
            return self._fallback_plan(query, columns)
        except Exception as exc:
            logger.warning("LLM planning failed; using fallback plan: %s", exc)
            return self._fallback_plan(query, columns)

        try:
            steps_json = self._extract_json(raw)
        except json.JSONDecodeError:
            logger.warning("Planner returned non-JSON; using fallback plan.")
            return self._fallback_plan(query, columns)
        steps: List[AgentStep] = []
        for item in steps_json:
            tool = item.get("tool", "")
            if tool:
                steps.append(AgentStep(tool=tool, params=item.get("params", {}) or {}, reason=item.get("reason", "")))
        return steps or self._fallback_plan(query, columns)

    def _extract_json(self, text: str) -> List[Dict[str, Any]]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def _fallback_plan(self, query: str, columns: List[str]) -> List[AgentStep]:
        q = query.lower()
        if "clean" in q or "missing" in q or "duplicate" in q:
            return [
                AgentStep(tool="data_quality_report", reason="Assess current data quality"),
                AgentStep(tool="clean_data", reason="Perform basic cleaning"),
                AgentStep(tool="summarize_dataset", reason="Summarize cleaned data"),
            ]
        if "quality" in q or "anomaly" in q:
            return [AgentStep(tool="data_quality_report", reason="Generate quality report and anomaly scan")]
        if "sql" in q or "query" in q:
            return [AgentStep(tool="run_sql", params={"query": "SELECT * FROM df LIMIT 20"}, reason="Run SQL analysis")]
        if "plot" in q or "chart" in q or "visual" in q:
            x_col = columns[0] if columns else ""
            return [
                AgentStep(tool="generate_plot", params={"x_col": x_col, "chart_type": "histogram"}, reason="Generate chart"),
                AgentStep(tool="generate_automatic_visualizations", reason="Generate additional dynamic visualizations"),
            ]
        return [
            AgentStep(tool="summarize_dataset", reason="Default dataset analysis"),
            AgentStep(tool="data_quality_report", reason="Include quality context"),
        ]
