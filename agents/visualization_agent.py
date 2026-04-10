from __future__ import annotations

from typing import List

from tools import DataTools, ToolResult


class VisualizationAgent:
    """Selects and builds Plotly charts from column roles."""

    NAME = "Visualization Agent"

    def run(self, tools: DataTools, max_charts: int = 3) -> List[ToolResult]:
        return tools.generate_pipeline_charts(max_charts=max_charts)
