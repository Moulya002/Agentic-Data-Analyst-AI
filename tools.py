from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px

from utils.chart_theme import human_axis_label, style_plotly_figure


def quality_summary_text(quality: Dict[str, Any]) -> str:
    """One readable paragraph for UI + exports from a quality_report dict (analyst tone)."""
    rows = int(quality.get("rows", 0) or 0)
    cols = int(quality.get("columns", 0) or 0)
    dups = int(quality.get("duplicate_rows", 0) or 0)
    miss = int(quality.get("total_missing_cells", 0) or 0)
    total_cells = max(1, rows * cols)
    miss_pct = 100.0 * miss / total_cells
    parts = [
        f"You are working with **{rows:,} rows** and **{cols} columns**.",
        f"About **{miss_pct:.1f}%** of all cells are empty ({miss:,} missing cells).",
    ]
    if rows > 0:
        dup_pct = 100.0 * dups / rows
        parts.append(f"**{dups:,} duplicate rows** (~{dup_pct:.1f}% of rows) — worth deduplicating before KPIs.")
    anomalies = quality.get("anomalies") or []
    if anomalies:
        parts.append(
            "**Outliers (IQR rule) show up in:** "
            + "; ".join(anomalies[:6])
            + (" …" if len(anomalies) > 6 else "")
            + " — investigate before treating them as errors."
        )
    mb = quality.get("missing_by_column") or {}
    if isinstance(mb, dict) and mb:
        worst = sorted(mb.items(), key=lambda x: -int(x[1] or 0))[:3]
        worst_txt = []
        for k, v in worst:
            col_pct = 100.0 * int(v) / max(1, rows)
            worst_txt.append(f"`{k}` (~{col_pct:.0f}% of rows affected, {int(v)} cells)")
        parts.append("**Missing values concentrate in:** " + ", ".join(worst_txt) + ".")
    return " ".join(parts)


@dataclass
class ToolResult:
    text: str
    table: Optional[pd.DataFrame] = None
    plotly_fig: Any = None
    generated_code: str = ""
    quality_report: Optional[Dict[str, Any]] = None


class DataTools:
    """
    Tool belt used by the agent.
    Each method returns a ToolResult to keep UI rendering simple.
    """

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.active_dataset_name: str = ""

    def load_csv(self, file_bytes: bytes, name: str = "uploaded.csv") -> ToolResult:
        self.df = pd.read_csv(BytesIO(file_bytes))
        self.datasets = {name: self.df.copy()}
        self.active_dataset_name = name
        return ToolResult(
            text=f"Loaded dataset with {self.df.shape[0]} rows and {self.df.shape[1]} columns.",
            table=self.df.head(10),
            generated_code=f"df = pd.read_csv('{name}')",
        )

    def load_multiple_csvs(self, files: List[Tuple[str, bytes]]) -> ToolResult:
        self.datasets = {}
        for name, file_bytes in files:
            self.datasets[name] = pd.read_csv(BytesIO(file_bytes))
        if not self.datasets:
            raise ValueError("No files provided.")

        names = list(self.datasets.keys())
        self.active_dataset_name = names[0]

        # Default behavior: vertically merge files with common columns.
        aligned = [df.copy() for df in self.datasets.values()]
        self.df = pd.concat(aligned, ignore_index=True, sort=False)
        return ToolResult(
            text=(
                f"Loaded {len(self.datasets)} datasets and created a merged view with "
                f"{self.df.shape[0]} rows x {self.df.shape[1]} columns."
            ),
            table=self.df.head(10),
            generated_code=(
                "# Multi-file ingestion\n"
                "dfs = [pd.read_csv(file) for file in files]\n"
                "df = pd.concat(dfs, ignore_index=True, sort=False)"
            ),
        )

    def suggest_join_candidates(self) -> List[Dict[str, Any]]:
        """
        Suggest likely join keys across uploaded datasets.
        Score is based on shared column names and overlap ratio.
        """
        if len(self.datasets) < 2:
            return []

        names = list(self.datasets.keys())
        suggestions: List[Dict[str, Any]] = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                left_name, right_name = names[i], names[j]
                left_df, right_df = self.datasets[left_name], self.datasets[right_name]
                common_cols = set(left_df.columns).intersection(set(right_df.columns))
                for col in common_cols:
                    left_vals = set(left_df[col].dropna().astype(str).unique().tolist())
                    right_vals = set(right_df[col].dropna().astype(str).unique().tolist())
                    if not left_vals or not right_vals:
                        continue
                    overlap = len(left_vals.intersection(right_vals))
                    denom = max(1, min(len(left_vals), len(right_vals)))
                    score = overlap / denom
                    if score >= 0.2:
                        suggestions.append(
                            {
                                "left_dataset": left_name,
                                "right_dataset": right_name,
                                "join_column": col,
                                "overlap_score": round(score, 3),
                                "overlap_values": overlap,
                            }
                        )
        suggestions.sort(key=lambda x: x["overlap_score"], reverse=True)
        return suggestions[:10]

    def apply_join_strategy(
        self,
        left_dataset: str,
        right_dataset: str,
        join_column: str,
        how: str = "inner",
    ) -> ToolResult:
        if left_dataset not in self.datasets or right_dataset not in self.datasets:
            raise ValueError("Selected datasets not found.")
        left_df = self.datasets[left_dataset]
        right_df = self.datasets[right_dataset]
        if join_column not in left_df.columns or join_column not in right_df.columns:
            raise ValueError(f"Join column '{join_column}' is missing in one dataset.")

        merged = pd.merge(
            left_df,
            right_df,
            on=join_column,
            how=how,
            suffixes=(f"_{left_dataset}", f"_{right_dataset}"),
        )
        self.df = merged
        self.active_dataset_name = f"{left_dataset}_{right_dataset}_{how}_join"
        return ToolResult(
            text=(
                f"Applied {how} join between `{left_dataset}` and `{right_dataset}` on `{join_column}`. "
                f"Result: {merged.shape[0]} rows x {merged.shape[1]} columns."
            ),
            table=merged.head(10),
            generated_code=(
                f"df = pd.merge(df_{left_dataset!r}, df_{right_dataset!r}, "
                f"on='{join_column}', how='{how}')"
            ),
        )

    def ensure_data(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("No dataset loaded. Upload a CSV first.")
        return self.df

    def quick_analyst_bullets(self) -> List[str]:
        """
        Short, numeric highlights for offline insight fallbacks and LLM grounding.
        """
        df = self.ensure_data()
        out: List[str] = []
        rows = len(df)
        if rows == 0:
            return out

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = [c for c in df.columns if c not in numeric_cols]

        if object_cols and numeric_cols:
            category_col = min(object_cols, key=lambda c: df[c].nunique(dropna=True))
            metric_col = numeric_cols[0]
            grouped = (
                df.groupby(category_col, dropna=False)[metric_col]
                .sum(min_count=1)
                .dropna()
                .sort_values(ascending=False)
            )
            if len(grouped) >= 1:
                top_val = grouped.iloc[0]
                total = grouped.sum()
                if total and abs(float(total)) > 1e-9:
                    pct = 100.0 * float(top_val) / float(total)
                    top_label = str(grouped.index[0])
                    out.append(
                        f"In `{category_col}`, **{top_label}** drives about **{pct:.0f}%** of total `{metric_col}` "
                        f"(share of summed `{metric_col}` across categories)."
                    )

        datetime_col = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_col = col
                break
            if df[col].dtype == object:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.7:
                    datetime_col = col
                    break

        if datetime_col and numeric_cols:
            trend_col = numeric_cols[0]
            tmp = df[[datetime_col, trend_col]].copy()
            tmp[datetime_col] = pd.to_datetime(tmp[datetime_col], errors="coerce")
            tmp = tmp.dropna(subset=[datetime_col, trend_col])
            if len(tmp) >= 3:
                monthly = tmp.groupby(tmp[datetime_col].dt.to_period("M"))[trend_col].mean()
                if len(monthly) >= 2:
                    last, prev = float(monthly.iloc[-1]), float(monthly.iloc[-2])
                    if prev != 0 and not (np.isnan(last) or np.isnan(prev)):
                        chg = 100.0 * (last - prev) / abs(prev)
                        direction = "up" if chg > 1 else "down" if chg < -1 else "flat"
                        out.append(
                            f"Latest month for `{trend_col}` vs prior month is **{direction}** "
                            f"(roughly **{chg:+.0f}%** change on a monthly average basis)."
                        )

        miss = df.isna().sum()
        worst_cols = miss[miss > 0].sort_values(ascending=False).head(2)
        if len(worst_cols) > 0:
            parts = []
            for col_name, cnt in worst_cols.items():
                pct_r = 100.0 * int(cnt) / max(1, rows)
                parts.append(f"`{col_name}` (~{pct_r:.0f}% of rows affected)")
            out.append("Missing values are most visible in: " + ", ".join(parts) + " — worth cleaning before modeling.")

        return out[:5]

    def summarize_dataset(self) -> ToolResult:
        df = self.ensure_data()
        summary = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": list(df.columns),
            "missing_values": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
        }
        dtype_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
        miss_pct = 100.0 * summary["missing_values"] / max(1, summary["rows"] * summary["columns"])
        dup_pct = 100.0 * summary["duplicates"] / max(1, summary["rows"])
        text = (
            f"**Scale:** {summary['rows']:,} records × {summary['columns']} fields.\n"
            f"**Data hygiene:** {summary['missing_values']:,} empty cells overall (~{miss_pct:.1f}% of the grid); "
            f"{summary['duplicates']:,} full duplicate rows (~{dup_pct:.1f}% of rows).\n"
            "**Implication:** clean before revenue or funnel metrics so totals stay credible."
        )
        return ToolResult(
            text=text,
            table=dtype_df,
            generated_code="summary = df.describe(include='all').transpose()",
        )

    def clean_data(self) -> ToolResult:
        df = self.ensure_data().copy()
        original_rows = len(df)
        duplicates = int(df.duplicated().sum())
        df = df.drop_duplicates()
        missing_before = int(df.isna().sum().sum())

        for col in df.select_dtypes(include=["number"]).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(exclude=["number"]).columns:
            df[col] = df[col].fillna("Unknown")

        missing_after = int(df.isna().sum().sum())
        self.df = df
        dup_pct = 100.0 * duplicates / max(1, original_rows)
        filled = missing_before - missing_after
        text = (
            f"**Deduplication:** removed **{duplicates}** repeat row(s) (~{dup_pct:.1f}% of the file).\n"
            f"**Imputation:** filled **{filled:,}** missing cells (numeric → median; text → “Unknown”).\n"
            f"**Resulting table:** **{len(df):,}** rows; **{missing_after:,}** cells still empty (if any, likely structural gaps)."
        )
        return ToolResult(
            text=text,
            table=df.head(10),
            generated_code=(
                "df = df.drop_duplicates()\n"
                "for c in df.select_dtypes(include=['number']).columns:\n"
                "    df[c] = df[c].fillna(df[c].median())\n"
                "for c in df.select_dtypes(exclude=['number']).columns:\n"
                "    df[c] = df[c].fillna('Unknown')"
            ),
        )

    def run_sql(self, query: str) -> ToolResult:
        df = self.ensure_data()
        con = duckdb.connect(database=":memory:")
        con.register("df", df)
        result = con.execute(query).df()
        return ToolResult(
            text=f"**Query result:** **{len(result)}** row(s) returned — use the table below for the exact figures.",
            table=result.head(100),
            generated_code=f"result = duckdb.sql(\"\"\"{query}\"\"\").df()",
        )

    def generate_plot(self, x_col: str, y_col: Optional[str] = None, chart_type: str = "histogram") -> ToolResult:
        df = self.ensure_data()
        if x_col not in df.columns:
            raise ValueError(f"Column not found: {x_col}")
        if y_col and y_col not in df.columns:
            raise ValueError(f"Column not found: {y_col}")

        chart_type = chart_type.lower()
        if chart_type == "scatter" and y_col:
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{human_axis_label(x_col)} vs {human_axis_label(y_col)}")
            style_plotly_figure(
                fig,
                title=f"{human_axis_label(x_col)} vs {human_axis_label(y_col)}",
                x_title=human_axis_label(x_col),
                y_title=human_axis_label(y_col),
            )
        elif chart_type == "line" and y_col:
            fig = px.line(df, x=x_col, y=y_col, title=f"Trend: {human_axis_label(y_col)}")
            style_plotly_figure(
                fig,
                title=f"Trend: {human_axis_label(y_col)} over {human_axis_label(x_col)}",
                x_title=human_axis_label(x_col),
                y_title=human_axis_label(y_col),
            )
        elif chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, title=f"{human_axis_label(x_col)} breakdown")
            y_lab = human_axis_label(y_col) if y_col else "Count"
            style_plotly_figure(
                fig,
                title=f"{human_axis_label(x_col)}" + (f" by {human_axis_label(y_col)}" if y_col else ""),
                x_title=human_axis_label(x_col),
                y_title=y_lab,
            )
        else:
            fig = px.histogram(df, x=x_col, nbins=30, title=f"Distribution of {human_axis_label(x_col)}")
            style_plotly_figure(
                fig,
                title=f"Distribution of {human_axis_label(x_col)}",
                x_title=human_axis_label(x_col),
                y_title="Frequency",
            )
        chart_fn = chart_type if chart_type in ["line", "bar", "scatter", "histogram"] else "histogram"
        y_part = f", y='{y_col}'" if y_col else ""
        return ToolResult(
            text=f"**Chart:** {chart_type} for `{x_col}`" + (f" vs `{y_col}`" if y_col else "") + " — see interactive view below.",
            plotly_fig=fig,
            generated_code=f"fig = px.{chart_fn}(df, x='{x_col}'{y_part})",
        )

    def data_quality_report(self) -> ToolResult:
        df = self.ensure_data()
        n_rows, n_cols = df.shape
        missing_per_col = df.isna().sum().sort_values(ascending=False)
        duplicates = int(df.duplicated().sum())

        anomaly_notes: List[str] = []
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = int(((series < low) | (series > high)).sum())
            if outliers > 0:
                anomaly_notes.append(f"{col}: {outliers} potential outliers")

        quality = {
            "rows": n_rows,
            "columns": n_cols,
            "duplicate_rows": duplicates,
            "total_missing_cells": int(df.isna().sum().sum()),
            "missing_by_column": missing_per_col[missing_per_col > 0].to_dict(),
            "anomalies": anomaly_notes,
        }
        quality_table = pd.DataFrame(
            {
                "column": list(missing_per_col.index),
                "missing_count": list(missing_per_col.values),
            }
        )
        dtype_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
        quality_table = quality_table.merge(dtype_df, on="column", how="left")
        quality_table = quality_table[["column", "dtype", "missing_count"]]

        return ToolResult(
            text=quality_summary_text(quality),
            table=quality_table,
            quality_report=quality,
            generated_code=(
                "missing_by_col = df.isna().sum()\n"
                "duplicates = df.duplicated().sum()\n"
                "# IQR-based outlier scan on numeric columns"
            ),
        )

    def analysis_profile(self) -> ToolResult:
        """Aggregations, describe stats, and top correlations (Analysis Agent)."""
        df = self.ensure_data()
        lines: List[str] = ["**Numeric deep-dive (for planning and storytelling):**"]
        num = df.select_dtypes(include=[np.number])
        table_parts: List[pd.DataFrame] = []

        if num.empty:
            lines.append("There are **no numeric KPI columns** here — focus on segment counts or encode categories before correlation work.")
        else:
            desc = num.describe().T.round(4)
            table_parts.append(desc.reset_index().rename(columns={"index": "column"}))
            lines.append(
                f"**{len(num.columns)} numeric field(s)** drive aggregates; the stats table shows typical level, spread, and extremes."
            )

            if len(num.columns) >= 2:
                corr = num.corr(numeric_only=True)
                pairs: List[Tuple[str, str, float]] = []
                ccols = corr.columns.tolist()
                for i in range(len(ccols)):
                    for j in range(i + 1, len(ccols)):
                        v = corr.iloc[i, j]
                        if pd.notna(v):
                            pairs.append((ccols[i], ccols[j], float(v)))
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                top = pairs[:4]
                if top:
                    human = []
                    for a, b, v in top:
                        strength = "strong" if abs(v) >= 0.6 else "moderate" if abs(v) >= 0.35 else "weak"
                        human.append(f"`{a}` and `{b}` move together ({strength} link, r≈{v:+.2f})")
                    lines.append("**Relationships:** " + "; ".join(human) + ".")

        cat_cols = [c for c in df.columns if c not in num.columns]
        if cat_cols:
            preview = ", ".join(f"`{c}`" for c in cat_cols[:8])
            tail = " …" if len(cat_cols) > 8 else ""
            lines.append(f"**Segmentation dimensions ({len(cat_cols)}):** {preview}{tail} — use these for breakdowns and bar charts.")

        summary_table = table_parts[0] if table_parts else pd.DataFrame({"note": ["No numeric describe available"]})
        return ToolResult(text="\n".join(lines), table=summary_table)

    def correlation_heatmap(self) -> ToolResult:
        """Plotly heatmap of numeric correlations."""
        df = self.ensure_data()
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            return ToolResult(text="Need at least two numeric columns to compare how KPIs move together.")
        corr = num.corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="How numeric KPIs correlate",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            labels=dict(x="Metric", y="Metric", color="Correlation"),
        )
        style_plotly_figure(fig, title="How numeric KPIs correlate (darker red = stronger positive)")
        fig.update_xaxes(side="bottom")
        return ToolResult(
            text="**Heatmap:** warmer cells mean two metrics rise/fall together; cooler blues mean they oppose — use this to prioritize joint drivers.",
            plotly_fig=fig,
        )

    def generate_pipeline_charts(self, max_charts: int = 3) -> List[ToolResult]:
        """
        Auto-select 2–3 charts: heatmap (if possible), trend, top categories, or histogram.
        """
        df = self.ensure_data()
        outputs: List[ToolResult] = []
        max_charts = max(1, min(max_charts, 6))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        datetime_col = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_col = col
                break
            if df[col].dtype == object:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.7:
                    datetime_col = col
                    break

        def add(result: ToolResult) -> None:
            if len(outputs) >= max_charts:
                return
            outputs.append(result)

        if len(numeric_cols) >= 2:
            add(self.correlation_heatmap())

        if datetime_col and numeric_cols and len(outputs) < max_charts:
            trend_col = numeric_cols[0]
            tmp = df.copy()
            tmp[datetime_col] = pd.to_datetime(tmp[datetime_col], errors="coerce")
            tmp = tmp.dropna(subset=[datetime_col, trend_col])
            if not tmp.empty:
                monthly = (
                    tmp.groupby(tmp[datetime_col].dt.to_period("M"))[trend_col]
                    .mean()
                    .reset_index()
                )
                monthly[datetime_col] = monthly[datetime_col].astype(str)
                fig = px.line(
                    monthly,
                    x=datetime_col,
                    y=trend_col,
                    markers=True,
                    title=f"Trend: {human_axis_label(trend_col)}",
                )
                style_plotly_figure(
                    fig,
                    title=f"Average {human_axis_label(trend_col)} over time",
                    x_title=human_axis_label(datetime_col),
                    y_title=human_axis_label(trend_col),
                )
                add(
                    ToolResult(
                        text=f"**Trend:** `{trend_col}` smoothed by period — use this to spot growth or slowdowns.",
                        plotly_fig=fig,
                    )
                )

        if object_cols and numeric_cols and len(outputs) < max_charts:
            category_col = min(object_cols, key=lambda c: df[c].nunique(dropna=True))
            metric_col = numeric_cols[0]
            grouped = (
                df.groupby(category_col, dropna=False)[metric_col]
                .sum(min_count=1)
                .reset_index()
                .sort_values(metric_col, ascending=False)
                .head(12)
            )
            fig = px.bar(
                grouped,
                x=category_col,
                y=metric_col,
                title=f"Top {human_axis_label(category_col)} by {human_axis_label(metric_col)}",
            )
            style_plotly_figure(
                fig,
                title=f"Where {human_axis_label(metric_col)} concentrates",
                x_title=human_axis_label(category_col),
                y_title=human_axis_label(metric_col),
            )
            add(
                ToolResult(
                    text=f"**Concentration:** `{metric_col}` is stacked into a few `{category_col}` values — the tall bars are your priority segments.",
                    plotly_fig=fig,
                )
            )

        if numeric_cols and len(outputs) < max_charts:
            target_col = numeric_cols[0]
            fig = px.histogram(df, x=target_col, nbins=30, title=f"Distribution of {human_axis_label(target_col)}")
            style_plotly_figure(
                fig,
                title=f"Distribution of {human_axis_label(target_col)}",
                x_title=human_axis_label(target_col),
                y_title="Count",
            )
            add(
                ToolResult(
                    text=f"**Shape:** `{target_col}` distribution — long tails or skew hint at discounting, caps, or data entry quirks.",
                    plotly_fig=fig,
                )
            )

        return outputs if outputs else [ToolResult(text="No charts generated (insufficient column types).")]

    def generate_automatic_visualizations(self) -> List[ToolResult]:
        """
        Build a small auto-dashboard from unknown datasets.
        Returns chart outputs that can be rendered directly in Streamlit.
        """
        df = self.ensure_data()
        outputs: List[ToolResult] = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Detect potential datetime columns dynamically.
        datetime_col = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_col = col
                break
            if df[col].dtype == object:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.7:
                    datetime_col = col
                    break

        # 1) Trend chart over time (if possible).
        if datetime_col and numeric_cols:
            trend_col = numeric_cols[0]
            tmp = df.copy()
            tmp[datetime_col] = pd.to_datetime(tmp[datetime_col], errors="coerce")
            tmp = tmp.dropna(subset=[datetime_col, trend_col])
            if not tmp.empty:
                monthly = (
                    tmp.groupby(tmp[datetime_col].dt.to_period("M"))[trend_col]
                    .mean()
                    .reset_index()
                )
                monthly[datetime_col] = monthly[datetime_col].astype(str)
                fig = px.line(
                    monthly,
                    x=datetime_col,
                    y=trend_col,
                    markers=True,
                    title=f"Trend: {human_axis_label(trend_col)}",
                )
                style_plotly_figure(
                    fig,
                    title=f"{human_axis_label(trend_col)} over time",
                    x_title=human_axis_label(datetime_col),
                    y_title=human_axis_label(trend_col),
                )
                outputs.append(
                    ToolResult(
                        text=f"**Trend signal:** `{trend_col}` over `{datetime_col}` — read slope and dips for seasonality or risk.",
                        plotly_fig=fig,
                    )
                )

        # 2) Top category performers (if possible).
        if object_cols and numeric_cols:
            category_col = min(object_cols, key=lambda c: df[c].nunique(dropna=True))
            metric_col = numeric_cols[0]
            grouped = (
                df.groupby(category_col, dropna=False)[metric_col]
                .sum(min_count=1)
                .reset_index()
                .sort_values(metric_col, ascending=False)
                .head(10)
            )
            fig = px.bar(
                grouped,
                x=category_col,
                y=metric_col,
                title=f"Top {human_axis_label(category_col)} by {human_axis_label(metric_col)}",
            )
            style_plotly_figure(
                fig,
                title=f"{human_axis_label(metric_col)} by {human_axis_label(category_col)}",
                x_title=human_axis_label(category_col),
                y_title=human_axis_label(metric_col),
            )
            outputs.append(
                ToolResult(
                    text=f"**Share of performance:** `{category_col}` drives `{metric_col}` — focus on the tallest bars first.",
                    plotly_fig=fig,
                )
            )

        # 3) Missing values profile.
        missing_df = df.isna().sum().reset_index()
        missing_df.columns = ["column", "missing_count"]
        missing_df = missing_df[missing_df["missing_count"] > 0].sort_values("missing_count", ascending=False).head(15)
        if not missing_df.empty:
            fig = px.bar(
                missing_df,
                x="column",
                y="missing_count",
                title="Missing values by column",
            )
            style_plotly_figure(
                fig,
                title="Where data is missing most",
                x_title="Column",
                y_title="Missing cells",
            )
            outputs.append(
                ToolResult(
                    text="**Data risk:** columns with the most gaps — fix these before forecasting or margin analysis.",
                    plotly_fig=fig,
                )
            )

        # 4) Numeric distribution using histogram.
        if numeric_cols:
            target_col = numeric_cols[0]
            fig = px.histogram(df, x=target_col, nbins=30, title=f"Distribution of {human_axis_label(target_col)}")
            style_plotly_figure(
                fig,
                title=f"Distribution of {human_axis_label(target_col)}",
                x_title=human_axis_label(target_col),
                y_title="Frequency",
            )
            outputs.append(
                ToolResult(
                    text=f"**Distribution:** `{target_col}` — check for skew, caps, or outliers that distort averages.",
                    plotly_fig=fig,
                )
            )

        return outputs
