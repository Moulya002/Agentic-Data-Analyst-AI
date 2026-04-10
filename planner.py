from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

    def plan(
        self,
        query: str,
        columns: List[str],
        memory_context: List[Dict[str, str]],
        column_dtypes: Optional[Dict[str, str]] = None,
    ) -> List[AgentStep]:
        dtypes = column_dtypes or {}
        if not self.llm.enabled:
            return self._fallback_plan(query, columns, dtypes)

        system_prompt = (
            "You are a planning agent for a data analyst system. "
            "Return ONLY valid JSON list of steps with keys: tool, params, reason.\n"
            "Allowed tools: summarize_dataset, clean_data, run_sql, generate_plot, data_quality_report, "
            "generate_automatic_visualizations, analysis_profile, correlation_heatmap, generate_pipeline_charts.\n"
            "For generate_pipeline_charts use params like {\"max_charts\": 3}.\n"
            "If SQL is needed, table name is df.\n"
            "Keep steps minimal (1-4)."
        )
        user_prompt = {
            "query": query,
            "columns": columns,
            "column_dtypes": dtypes,
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
            return self._fallback_plan(query, columns, dtypes)
        except APIStatusError as exc:
            logger.warning("Groq API error (planner); using fallback plan: %s", exc)
            return self._fallback_plan(query, columns, dtypes)
        except Exception as exc:
            logger.warning("LLM planning failed; using fallback plan: %s", exc)
            return self._fallback_plan(query, columns, dtypes)

        try:
            steps_json = self._extract_json(raw)
        except json.JSONDecodeError:
            logger.warning("Planner returned non-JSON; using fallback plan.")
            return self._fallback_plan(query, columns, dtypes)
        steps: List[AgentStep] = []
        for item in steps_json:
            tool = item.get("tool", "")
            if tool:
                steps.append(AgentStep(tool=tool, params=item.get("params", {}) or {}, reason=item.get("reason", "")))
        return steps or self._fallback_plan(query, columns, dtypes)

    def _extract_json(self, text: str) -> List[Dict[str, Any]]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise

    @staticmethod
    def _quote_sql_col(name: str) -> str:
        escaped = name.replace('"', '""')
        return f'"{escaped}"'

    @staticmethod
    def _is_numeric_dtype(dtype: str) -> bool:
        d = (dtype or "").lower()
        return any(x in d for x in ("int", "float", "double", "decimal", "uint"))

    @staticmethod
    def _mentioned_columns(query: str, columns: List[str]) -> List[str]:
        ql = query.lower()
        found: List[str] = []
        for col in sorted(columns, key=len, reverse=True):
            if col.lower() in ql:
                found.append(col)
        return found

    def _fallback_plan(self, query: str, columns: List[str], column_dtypes: Dict[str, str]) -> List[AgentStep]:
        q = query.lower()
        dtypes = column_dtypes or {}
        mentioned = self._mentioned_columns(query, columns) if columns else []

        def dtype_of(col: str) -> str:
            return dtypes.get(col, "")

        # Auto-insight prompt from agent.generate_automatic_insights
        if "automatically analyze" in q or "key business insights" in q:
            return [
                AgentStep(tool="summarize_dataset", reason="Dataset overview"),
                AgentStep(tool="data_quality_report", reason="Quality and anomalies"),
                AgentStep(tool="generate_automatic_visualizations", reason="Auto charts"),
            ]
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
        if "correlat" in q or "heatmap" in q:
            return [
                AgentStep(tool="analysis_profile", reason="Correlation and numeric profile"),
                AgentStep(tool="correlation_heatmap", reason="Visualize correlation matrix"),
            ]
        if "aggregat" in q or "trend" in q or "summary stat" in q:
            return [
                AgentStep(tool="summarize_dataset", reason="Dataset shape and types"),
                AgentStep(tool="analysis_profile", reason="Numeric aggregates and correlations"),
            ]

        if re.search(r"\b(how many rows|row count|number of rows|how many records)\b", q):
            return [
                AgentStep(
                    tool="run_sql",
                    params={"query": "SELECT COUNT(*) AS row_count FROM df"},
                    reason="Answer row count from the table",
                )
            ]

        if any(k in q for k in ("preview", "first rows", "head", "sample rows", "show me the data")):
            return [
                AgentStep(
                    tool="run_sql",
                    params={"query": "SELECT * FROM df LIMIT 15"},
                    reason="Preview raw rows",
                )
            ]

        for col in mentioned:
            qc = self._quote_sql_col(col)
            dt = dtype_of(col)
            numeric = self._is_numeric_dtype(dt)
            if numeric and any(w in q for w in ("average", "mean", "avg")):
                return [
                    AgentStep(
                        tool="run_sql",
                        params={"query": f"SELECT AVG(CAST({qc} AS DOUBLE)) AS avg_{col.replace(' ', '_')[:40]} FROM df"},
                        reason=f"Compute average of {col}",
                    )
                ]
            if numeric and "sum" in q:
                return [
                    AgentStep(
                        tool="run_sql",
                        params={"query": f"SELECT SUM(CAST({qc} AS DOUBLE)) AS sum_{col.replace(' ', '_')[:40]} FROM df"},
                        reason=f"Compute sum of {col}",
                    )
                ]
            if numeric and ("maximum" in q or re.search(r"\bmax\b", q)):
                return [
                    AgentStep(
                        tool="run_sql",
                        params={"query": f"SELECT MAX(CAST({qc} AS DOUBLE)) AS max_{col.replace(' ', '_')[:40]} FROM df"},
                        reason=f"Maximum of {col}",
                    )
                ]
            if numeric and ("minimum" in q or re.search(r"\bmin\b", q)):
                return [
                    AgentStep(
                        tool="run_sql",
                        params={"query": f"SELECT MIN(CAST({qc} AS DOUBLE)) AS min_{col.replace(' ', '_')[:40]} FROM df"},
                        reason=f"Minimum of {col}",
                    )
                ]
            if not numeric and any(w in q for w in ("top", "frequent", "common", "breakdown", "distribution", "count by")):
                return [
                    AgentStep(
                        tool="run_sql",
                        params={
                            "query": (
                                f"SELECT {qc} AS value, COUNT(*) AS n FROM df GROUP BY 1 ORDER BY n DESC NULLS LAST LIMIT 20"
                            )
                        },
                        reason=f"Frequency breakdown for {col}",
                    )
                ]

        if mentioned and self._is_numeric_dtype(dtype_of(mentioned[0])):
            c0 = mentioned[0]
            return [
                AgentStep(
                    tool="generate_plot",
                    params={"x_col": c0, "chart_type": "histogram"},
                    reason=f"Visualize distribution of {c0} mentioned in the question",
                ),
                AgentStep(tool="analysis_profile", reason="Numeric context for follow-up"),
            ]

        if mentioned:
            c0 = mentioned[0]
            return [
                AgentStep(
                    tool="run_sql",
                    params={
                        "query": (
                            f"SELECT {self._quote_sql_col(c0)} AS value, COUNT(*) AS n FROM df "
                            f"GROUP BY 1 ORDER BY n DESC NULLS LAST LIMIT 20"
                        )
                    },
                    reason=f"Top values for {c0} referenced in the question",
                )
            ]

        # Default: broader profile than the old summarize+quality pair (which looked identical for every question).
        return [
            AgentStep(tool="summarize_dataset", reason="Dataset shape, columns, and dtypes"),
            AgentStep(tool="analysis_profile", reason="Numeric summaries and correlations"),
            AgentStep(
                tool="run_sql",
                params={"query": "SELECT * FROM df LIMIT 8"},
                reason="Concrete row preview to ground the answer",
            ),
        ]
