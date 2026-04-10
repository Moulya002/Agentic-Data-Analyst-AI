"""Consistent Plotly styling for Streamlit (dark, readable, portfolio-ready)."""

from __future__ import annotations

from typing import Any, Optional


def human_axis_label(name: str) -> str:
    return str(name).replace("_", " ").strip().title()


def style_plotly_figure(
    fig: Any,
    *,
    title: Optional[str] = None,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
) -> Any:
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16)))
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)
    fig.update_layout(
        template="plotly_dark",
        height=480,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(148, 163, 184, 0.15)",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(148, 163, 184, 0.15)",
        zeroline=False,
    )
    return fig
