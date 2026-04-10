from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from executor import ExecutionAgent
from insight_agent import InsightAgent
from memory import MemoryStore
from planner import AgentStep, PlannerAgent
from tools import DataTools, ToolResult
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    final_answer: str
    refined_answer: str
    steps: List[AgentStep]
    tool_outputs: List[ToolResult]
    plan_summary: str = ""
    generated_code_blocks: List[str] = field(default_factory=list)
    data_quality_report: Dict[str, object] = field(default_factory=dict)


class AgenticDataAnalyst:
    def __init__(self, tools: DataTools, memory: MemoryStore):
        self.tools = tools
        self.memory = memory
        self.planner = PlannerAgent()
        self.executor = ExecutionAgent(tools)
        self.insight_agent = InsightAgent()

    def run(self, user_query: str, explanation_mode: str = "simple") -> AgentResponse:
        self.memory.add("user", user_query)
        df = self.tools.ensure_data()
        steps = self.planner.plan(
            user_query,
            list(df.columns),
            self.memory.recent(),
            column_dtypes={c: str(df[c].dtype) for c in df.columns},
        )
        outputs = self.executor.execute_plan(steps)
        initial_answer = self.insight_agent.summarize(user_query, steps, outputs, mode=explanation_mode)
        refined_answer = self.insight_agent.reflect_and_refine(
            user_query=user_query,
            initial_answer=initial_answer,
            steps=steps,
            outputs=outputs,
            mode=explanation_mode,
        )
        plan_summary = self._build_plan_summary(steps)
        self.memory.add("assistant", refined_answer)
        generated_code = [o.generated_code for o in outputs if o.generated_code]
        quality_reports = [o.quality_report for o in outputs if o.quality_report]
        return AgentResponse(
            final_answer=initial_answer,
            refined_answer=refined_answer,
            steps=steps,
            tool_outputs=outputs,
            plan_summary=plan_summary,
            generated_code_blocks=generated_code,
            data_quality_report=quality_reports[-1] if quality_reports else {},
        )

    def generate_automatic_insights(self) -> List[str]:
        """
        Insights without regenerating charts (for narrow code paths).
        Prefer generate_insights_and_charts for upload / demo flows.
        """
        self.tools.ensure_data()
        q = self.tools.data_quality_report()
        analysis = self.tools.analysis_profile()
        highlights = self.tools.quick_analyst_bullets()
        return self.insight_agent.pipeline_insights(
            quality_report=q.quality_report or {},
            clean_summary=(
                "No separate cleaning step in this path — use **Run full analysis** "
                "for dedupe and imputation before trusting rolled-up KPIs."
            ),
            analysis_summary=analysis.text,
            chart_descriptions=[],
            mode="simple",
            computed_highlights=highlights,
        )

    def generate_automatic_visualizations(self) -> List[ToolResult]:
        """
        Generate autonomous charts right after dataset upload.
        """
        try:
            return self.tools.generate_automatic_visualizations()
        except Exception as exc:
            logger.warning("Automatic visualization generation failed: %s", exc)
            return [ToolResult(text=f"Automatic visualization failed: {exc}")]

    def generate_insights_and_charts(
        self,
        df: pd.DataFrame,
        on_step: Optional[Callable[[str], None]] = None,
    ) -> Tuple[List[str], List[ToolResult], Dict[str, object], Optional[pd.DataFrame]]:
        """
        Quality scan → profile → charts → analyst-style insights (pipeline_insights).
        Returns quality metadata so the UI avoids a duplicate quality pass.
        """
        def step(msg: str) -> None:
            if on_step is not None:
                on_step(msg)

        self.tools.df = df.copy()
        clean_summary = (
            "Dataset is shown **as uploaded** (load does not auto-clean). "
            "Click **Run full analysis** to dedupe and impute in one pass."
        )
        step("📋 Step 1 · Assessing data quality — missing cells, duplicates, outliers…")
        q = self.tools.data_quality_report()
        step("🔎 Step 2 · Detecting patterns — numeric spreads and strongest correlations…")
        analysis = self.tools.analysis_profile()
        highlights = self.tools.quick_analyst_bullets()
        step("📊 Step 3 · Creating visualizations…")
        charts = self.generate_automatic_visualizations()
        chart_desc = [(c.text or "").replace("**", "").strip() for c in charts if c.text]
        step("💡 Step 4 · Writing insights in plain analyst language…")
        insights = self.insight_agent.pipeline_insights(
            quality_report=q.quality_report or {},
            clean_summary=clean_summary,
            analysis_summary=analysis.text,
            chart_descriptions=chart_desc,
            mode="simple",
            computed_highlights=highlights,
        )
        return insights, charts, q.quality_report or {}, q.table

    def run_full_analysis(
        self,
        on_step: Optional[Callable[[str], None]] = None,
    ) -> Tuple[List[str], List[ToolResult], Dict[str, object], List[str]]:
        """
        One-click: clean → quality snapshot → profile → charts → insights → recommendations.
        """
        def step(msg: str) -> None:
            if on_step is not None:
                on_step(msg)

        self.tools.ensure_data()
        step("🧹 Step 1 · Cleaning data — dedupe rows and fill missing values…")
        clean_res = self.tools.clean_data()
        step("📋 Step 2 · Recording data quality after cleaning…")
        q = self.tools.data_quality_report()
        step("🔎 Step 3 · Detecting patterns across columns…")
        analysis = self.tools.analysis_profile()
        highlights = self.tools.quick_analyst_bullets()
        step("📊 Step 4 · Creating visualizations…")
        charts = self.generate_automatic_visualizations()
        chart_desc = [(c.text or "").replace("**", "").strip() for c in charts if c.text]
        step("💡 Step 5 · Synthesizing insights…")
        insights = self.insight_agent.pipeline_insights(
            quality_report=q.quality_report or {},
            clean_summary=clean_res.text,
            analysis_summary=analysis.text,
            chart_descriptions=chart_desc,
            mode="simple",
            computed_highlights=highlights,
        )
        step("📌 Step 6 · Drafting business recommendations…")
        recs = self.insight_agent.business_recommendations(insights, q.quality_report or {})
        return insights, charts, q.quality_report or {}, recs

    def business_recommendations(self, insights: List[str], quality_report: Optional[Dict[str, object]] = None) -> List[str]:
        return self.insight_agent.business_recommendations(insights, quality_report or {})

    def _build_plan_summary(self, steps: List[AgentStep]) -> str:
        lines = ["Plan:"]
        for idx, step in enumerate(steps, start=1):
            lines.append(f"{idx}. {step.tool} - {step.reason or 'Execute analysis step'}")
        return "\n".join(lines)
