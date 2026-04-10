from __future__ import annotations

from tools import DataTools, ToolResult


class AnalysisAgent:
    """Aggregations, trends, and correlation structure."""

    NAME = "Analysis Agent"

    def run(self, tools: DataTools) -> ToolResult:
        return tools.analysis_profile()
