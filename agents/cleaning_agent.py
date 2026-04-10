from __future__ import annotations

from tools import DataTools, ToolResult


class CleaningAgent:
    """Handles missing values, duplicates, and basic type hygiene."""

    NAME = "Data Cleaning Agent"

    def run(self, tools: DataTools) -> ToolResult:
        return tools.clean_data()
