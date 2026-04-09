from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

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
        steps = self.planner.plan(user_query, list(df.columns), self.memory.recent())
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
        # Reuse multi-agent flow for autonomous analyst behavior.
        response = self.run(
            "Automatically analyze this dataset and provide 4-6 key business insights "
            "covering trends, top categories/products, anomalies/missing data, and recommendations.",
            explanation_mode="simple",
        )
        lines = [line.strip("- ").strip() for line in response.refined_answer.splitlines() if line.strip()]
        bullets = [line for line in lines if len(line) > 20][:6]
        return bullets or [response.refined_answer]

    def generate_automatic_visualizations(self) -> List[ToolResult]:
        """
        Generate autonomous charts right after dataset upload.
        """
        try:
            return self.tools.generate_automatic_visualizations()
        except Exception as exc:
            logger.warning("Automatic visualization generation failed: %s", exc)
            return [ToolResult(text=f"Automatic visualization failed: {exc}")]

    def generate_insights_and_charts(self, df: pd.DataFrame) -> tuple[List[str], List[ToolResult]]:
        """
        Required combined helper to produce both insights and charts.
        Accepts df explicitly to keep function contract clear.
        """
        self.tools.df = df.copy()
        insights = self.generate_automatic_insights()
        charts = self.generate_automatic_visualizations()
        return insights, charts

    def _build_plan_summary(self, steps: List[AgentStep]) -> str:
        lines = ["Plan:"]
        for idx, step in enumerate(steps, start=1):
            lines.append(f"{idx}. {step.tool} - {step.reason or 'Execute analysis step'}")
        return "\n".join(lines)
