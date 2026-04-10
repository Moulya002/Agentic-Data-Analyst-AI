import streamlit as st

from agent import AgenticDataAnalyst
from memory import MemoryStore
from tools import DataTools, quality_summary_text
from utils.config import settings
from utils.llm_client import GroqLLM

st.set_page_config(page_title="Agentic Data Analyst AI", page_icon="🤖", layout="wide")

# Portfolio polish: calmer typography + spacing
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; }
    h2 { margin-top: 1.5rem; margin-bottom: 0.5rem; font-weight: 600; }
    div[data-testid="stExpander"] details { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

REASONING_BLURBS = {
    "summarize_dataset": ("📋", "Scan dataset size, columns, dtypes, missing cells, and duplicate rows."),
    "clean_data": ("🧹", "Clean the table — dedupe rows and impute gaps so KPIs stay credible."),
    "data_quality_report": ("🔎", "Quantify missing values, duplicates, and outlier risk on key numerics."),
    "run_sql": ("📐", "Run a focused SQL cut to answer the question with exact rows."),
    "generate_plot": ("📊", "Build a targeted chart for the columns in scope."),
    "generate_automatic_visualizations": ("📈", "Spin up auto-selected charts (trend, category, missingness, distribution)."),
    "generate_pipeline_charts": ("📈", "Generate pipeline charts (heatmap, trend, bars, or histogram)."),
    "analysis_profile": ("🧠", "Profile numeric fields — typical ranges and strongest correlations."),
    "correlation_heatmap": ("🔥", "Show how numeric metrics move together in one heatmap."),
}


def friendly_plan_text(steps) -> str:
    lines = []
    for i, step in enumerate(steps, start=1):
        icon, blurb = REASONING_BLURBS.get(step.tool, ("⚙️", step.reason or "Execute an analysis step."))
        lines.append(f"Step {i}: {icon} {blurb}")
        if step.reason and step.tool in REASONING_BLURBS:
            lines.append(f"    → {step.reason}")
    return "\n".join(lines)


def build_export_report() -> str:
    lines = [
        "Agentic Data Analyst — Insights Report",
        "=" * 44,
        "",
        "KEY INSIGHTS",
        "-" * 12,
        "",
    ]
    for item in st.session_state.get("auto_insights") or []:
        lines.append(f"• {item}")
    lines.extend(["", "RECOMMENDATIONS", "-" * 16, ""])
    for r in st.session_state.get("business_recommendations") or []:
        lines.append(f"• {r}")
    lines.extend(["", "DATA QUALITY (summary)", "-" * 22, ""])
    qr = st.session_state.get("quality_report") or {}
    if qr:
        lines.append(f"Rows: {qr.get('rows', '—')} | Columns: {qr.get('columns', '—')}")
        lines.append(f"Duplicate rows: {qr.get('duplicate_rows', '—')} | Missing cells: {qr.get('total_missing_cells', '—')}")
    else:
        lines.append("No quality snapshot in this export.")
    return "\n".join(lines)


st.title("Agentic Data Analyst AI")
st.caption("Multi-agent pipeline: plan → tools → insights → charts — built for demos and real analysis.")

if "tools" not in st.session_state:
    st.session_state.tools = DataTools()
if "memory" not in st.session_state:
    st.session_state.memory = MemoryStore(
        max_messages=settings.max_memory_messages,
        storage_path=settings.memory_file_path,
    )
if "agent" not in st.session_state:
    st.session_state.agent = AgenticDataAnalyst(st.session_state.tools, st.session_state.memory)
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
if "auto_insights" not in st.session_state:
    st.session_state.auto_insights = []
if "auto_visuals" not in st.session_state:
    st.session_state.auto_visuals = []
if "quality_report" not in st.session_state:
    st.session_state.quality_report = {}
if "quality_table" not in st.session_state:
    st.session_state.quality_table = None
if "business_recommendations" not in st.session_state:
    st.session_state.business_recommendations = []
if "join_suggestions" not in st.session_state:
    st.session_state.join_suggestions = []
if "llm_health" not in st.session_state:
    st.session_state.llm_health = {}
if "pending_chat" not in st.session_state:
    st.session_state.pending_chat = None

with st.sidebar:
    st.header("Configuration")
    st.write("Model:", settings.groq_model)
    if not settings.use_llm:
        st.info("**LLM off** — `USE_LLM=false`. Planning and wording use **local rules** (no API calls).")
    elif settings.llm_enabled:
        st.success("**LLM on** — Groq powers planning and narrative polish.")
    else:
        st.info(
            "**Running in fallback mode** — rule-based planning and insights (no Groq key or placeholder key). "
            "Everything still works for demos; add a real `GROQ_API_KEY` in `.env` when you want LLM wording."
        )
    with st.expander("Optional: enable Groq LLM"):
        st.markdown(
            """
1. Key from [console.groq.com](https://console.groq.com/) (`gsk_…`).
2. Put in **`.env`**: `GROQ_API_KEY=gsk_…` — restart Streamlit.
3. Or keep **`USE_LLM=false`** for fully offline runs.
            """
        )
    show_reasoning = st.toggle("Show agent reasoning", value=settings.enable_reasoning_trace)
    explanation_mode = st.radio(
        "Explanation mode",
        ["Simple Explanation", "Technical Explanation"],
        horizontal=False,
    )
    if st.button("Clear memory"):
        st.session_state.memory.clear()
        st.session_state.chat_log = []
        st.success("Memory cleared.")

    st.divider()
    st.subheader("LLM Health Check")
    if st.button("Test Groq Connection"):
        checker = GroqLLM()
        st.session_state.llm_health = checker.health_check()
    if st.session_state.llm_health:
        health = st.session_state.llm_health
        if health.get("ok"):
            st.success(f"Connected in {health.get('latency_ms')} ms")
        else:
            st.error(f"Unavailable ({health.get('latency_ms')} ms): {health.get('error')}")

st.subheader("Dataset")
uploaded_files = st.file_uploader("Upload one or multiple CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Load dataset"):
        if len(uploaded_files) == 1:
            file = uploaded_files[0]
            load_result = st.session_state.tools.load_csv(file.getvalue(), name=file.name)
        else:
            files = [(file.name, file.getvalue()) for file in uploaded_files]
            load_result = st.session_state.tools.load_multiple_csvs(files)
            st.session_state.join_suggestions = st.session_state.tools.suggest_join_candidates()

        st.success(load_result.text)
        with st.spinner("Running auto insights and charts…"):
            try:
                st.session_state.auto_insights, st.session_state.auto_visuals = (
                    st.session_state.agent.generate_insights_and_charts(st.session_state.tools.df)
                )
            except Exception as exc:
                st.session_state.auto_insights = [
                    f"Insight step hit an error (check `.env` / Groq). Detail: {exc}"
                ]
                st.session_state.auto_visuals = st.session_state.agent.generate_automatic_visualizations()
                st.warning("Used rule-based fallbacks for this step; chat and charts may still run.")
            quality_result = st.session_state.tools.data_quality_report()
            st.session_state.quality_report = quality_result.quality_report or {}
            st.session_state.quality_table = quality_result.table
            st.session_state.business_recommendations = st.session_state.agent.business_recommendations(
                st.session_state.auto_insights,
                st.session_state.quality_report,
            )
        st.success("Load complete.")

df = st.session_state.tools.df

if df is not None:
    st.markdown("### Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    with st.expander("Preview (first rows)", expanded=False):
        st.dataframe(df.head(12), use_container_width=True)

    st.divider()
    st.markdown("### One-click full analysis")
    st.caption("Clean → refresh quality → regenerate insights & charts → refresh recommendations.")
    if st.button("Run full analysis", type="primary", help="End-to-end pipeline on the loaded table"):
        with st.spinner("Running full analysis pipeline…"):
            try:
                ins, charts, qr, recs = st.session_state.agent.run_full_analysis()
                st.session_state.auto_insights = ins
                st.session_state.auto_visuals = charts
                st.session_state.quality_report = qr
                qtab = st.session_state.tools.data_quality_report()
                st.session_state.quality_table = qtab.table
                st.session_state.business_recommendations = recs
                st.success("Full analysis complete.")
            except Exception as exc:
                st.error(f"Full analysis failed: {exc}")

if st.session_state.join_suggestions:
    st.subheader("Suggested cross-file joins")
    with st.expander("Review and apply suggested join", expanded=False):
        suggestion_labels = [
            f"{s['left_dataset']} <> {s['right_dataset']} on {s['join_column']} (score={s['overlap_score']})"
            for s in st.session_state.join_suggestions
        ]
        selected_idx = st.selectbox(
            "Join suggestion",
            options=list(range(len(suggestion_labels))),
            format_func=lambda i: suggestion_labels[i],
        )
        join_type = st.selectbox("Join type", ["inner", "left", "right", "outer"], index=0)
        if st.button("Apply selected join"):
            sel = st.session_state.join_suggestions[selected_idx]
            join_result = st.session_state.tools.apply_join_strategy(
                left_dataset=sel["left_dataset"],
                right_dataset=sel["right_dataset"],
                join_column=sel["join_column"],
                how=join_type,
            )
            st.success(join_result.text)
            st.dataframe(join_result.table, use_container_width=True)

if st.session_state.quality_report or st.session_state.quality_table is not None:
    st.markdown("### Data quality report")
    if st.session_state.quality_table is not None:
        st.dataframe(st.session_state.quality_table, use_container_width=True, hide_index=True)
    st.markdown(quality_summary_text(st.session_state.quality_report or {}))
    with st.expander("Raw quality JSON", expanded=False):
        st.json(st.session_state.quality_report)

if st.session_state.auto_insights:
    st.markdown("### Auto insights")
    for item in st.session_state.auto_insights[:8]:
        st.markdown(f"- {item}")

if st.session_state.business_recommendations:
    st.markdown("### Recommendations")
    st.caption("Action items recruiters and stakeholders can act on.")
    for rec in st.session_state.business_recommendations:
        st.markdown(f"- {rec}")

if st.session_state.auto_visuals:
    st.markdown("### Charts")
    for visual in st.session_state.auto_visuals:
        st.markdown(f"*{visual.text}*")
        if visual.plotly_fig is not None:
            st.plotly_chart(visual.plotly_fig, use_container_width=True)

if df is not None:
    st.divider()
    st.markdown("### Export")
    exp1, exp2 = st.columns(2)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    exp1.download_button(
        label="Download cleaned dataset (CSV)",
        data=csv_bytes,
        file_name="dataset_export.csv",
        mime="text/csv",
        help="Current in-memory table (includes any cleaning you ran).",
    )
    report_txt = build_export_report().encode("utf-8")
    exp2.download_button(
        label="Download insights report (TXT)",
        data=report_txt,
        file_name="insights_report.txt",
        mime="text/plain",
    )

    st.markdown("### Chat with your data")
    st.caption("Quick prompts — or type your own question below.")
    q1, q2, q3, q4 = st.columns(4)
    if q1.button("Clean data"):
        st.session_state.pending_chat = (
            "Clean duplicate rows, fill missing values (numeric median, text as Unknown), "
            "then summarize what changed in plain business language."
        )
    if q2.button("Generate insights"):
        st.session_state.pending_chat = (
            "Give me 5 concise business insights: where value concentrates, risks, missing data, and what to watch next."
        )
    if q3.button("Create charts"):
        st.session_state.pending_chat = (
            "Pick the most useful charts for this dataset, generate them, and explain what each chart implies."
        )
    if q4.button("Data quality"):
        st.session_state.pending_chat = (
            "Run a data quality assessment: missing values, duplicates, outliers, and which columns to fix first."
        )

    user_query = st.chat_input("Ask anything about your table…")
    if user_query:
        st.session_state.pending_chat = None
        query_to_run = user_query
    else:
        query_to_run = st.session_state.pop("pending_chat", None)

    if query_to_run:
        st.session_state.chat_log.append({"role": "user", "content": query_to_run})
        mode = "technical" if explanation_mode == "Technical Explanation" else "simple"
        try:
            response = st.session_state.agent.run(query_to_run, explanation_mode=mode)
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            response = None
        if response is not None:
            st.session_state.chat_log.append(
                {"role": "assistant", "content": response.refined_answer, "response_obj": response}
            )
            st.session_state.quality_report = response.data_quality_report or st.session_state.quality_report

    for msg in st.session_state.chat_log:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and "response_obj" in msg:
                response_obj = msg["response_obj"]
                with st.expander("Planned workflow", expanded=False):
                    st.text(friendly_plan_text(response_obj.steps) or "Plan not available.")
                if show_reasoning:
                    with st.expander("Agent reasoning steps", expanded=False):
                        st.markdown(
                            "_Each step below maps to a concrete tool run. Expand **Planned workflow** for a short narrative._"
                        )
                        for i, step in enumerate(response_obj.steps, start=1):
                            icon, blurb = REASONING_BLURBS.get(
                                step.tool, ("⚙️", step.reason or step.tool.replace("_", " "))
                            )
                            st.markdown(f"**Step {i}** &nbsp; {icon} **{blurb}**")
                            st.caption(f"Tool: `{step.tool}` · {step.reason or '—'}")
                            if step.params:
                                st.json(step.params, expanded=False)

                for out in response_obj.tool_outputs:
                    if out.table is not None:
                        st.dataframe(out.table, use_container_width=True)
                    if out.plotly_fig is not None:
                        st.plotly_chart(out.plotly_fig, use_container_width=True)

else:
    st.info("Upload a CSV and click **Load dataset** to begin.")
