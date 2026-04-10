from __future__ import annotations

from typing import List

from planner import AgentStep
from tools import DataTools, ToolResult
from utils.retry import with_retry


class ExecutionAgent:
    """
    Execution Agent:
    - receives a plan
    - executes each step using tool functions
    """

    def __init__(self, tools: DataTools):
        self.tools = tools

    @with_retry
    def execute_step(self, step: AgentStep) -> ToolResult:
        if step.tool == "summarize_dataset":
            return self.tools.summarize_dataset()
        if step.tool == "clean_data":
            return self.tools.clean_data()
        if step.tool == "run_sql":
            query = step.params.get("query", "SELECT * FROM df LIMIT 20")
            return self.tools.run_sql(query)
        if step.tool == "generate_plot":
            return self.tools.generate_plot(
                x_col=step.params.get("x_col"),
                y_col=step.params.get("y_col"),
                chart_type=step.params.get("chart_type", "histogram"),
            )
        if step.tool == "data_quality_report":
            return self.tools.data_quality_report()
        if step.tool == "analysis_profile":
            return self.tools.analysis_profile()
        if step.tool == "correlation_heatmap":
            return self.tools.correlation_heatmap()
        if step.tool == "generate_pipeline_charts":
            n = int(step.params.get("max_charts", 3))
            charts = self.tools.generate_pipeline_charts(max_charts=n)
            return ToolResult(text=f"Generated {len(charts)} pipeline chart(s).")
        if step.tool == "generate_automatic_visualizations":
            visuals = self.tools.generate_automatic_visualizations()
            # Wrap as a textual result while execution_plan appends visuals separately.
            return ToolResult(text=f"Generated {len(visuals)} automatic visualizations.")
        raise ValueError(f"Unknown tool: {step.tool}")

    def execute_plan(self, steps: List[AgentStep]) -> List[ToolResult]:
        outputs: List[ToolResult] = []
        for step in steps:
            try:
                if step.tool == "generate_automatic_visualizations":
                    outputs.extend(self.tools.generate_automatic_visualizations())
                elif step.tool == "generate_pipeline_charts":
                    outputs.extend(self.tools.generate_pipeline_charts(max_charts=int(step.params.get("max_charts", 3))))
                else:
                    outputs.append(self.execute_step(step))
            except Exception as exc:
                outputs.append(ToolResult(text=f"Step failed ({step.tool}): {exc}"))
        return outputs
