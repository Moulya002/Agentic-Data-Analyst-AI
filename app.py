import streamlit as st

from agent import AgenticDataAnalyst
from memory import MemoryStore
from tools import DataTools
from utils.config import settings
from utils.llm_client import GroqLLM

st.set_page_config(page_title="Agentic Data Analyst AI", page_icon="🤖", layout="wide")

st.title("Agentic Data Analyst AI")
st.caption("Production-style multi-agent analyst with LLM planning, reflection, and interactive visuals.")

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
if "join_suggestions" not in st.session_state:
    st.session_state.join_suggestions = []
if "llm_health" not in st.session_state:
    st.session_state.llm_health = {}

with st.sidebar:
    st.header("Configuration")
    st.write("Model:", settings.groq_model)
    if not settings.use_llm:
        st.info("**LLM off** — `USE_LLM=false`. All analysis uses local rules (no API calls).")
    elif settings.llm_enabled:
        st.success("**LLM on** — Groq will be used for planning and wording.")
    else:
        st.warning(
            "**LLM unavailable** — missing or placeholder `GROQ_API_KEY`. "
            "App runs in **offline mode** (same as no LLM)."
        )
    with st.expander("If Groq / API keys fail"):
        st.markdown(
            """
1. Create a key at [console.groq.com](https://console.groq.com/) (starts with `gsk_`).
2. Put it only in **`.env`**: `GROQ_API_KEY=gsk_...` (no quotes, no spaces).
3. **Restart** Streamlit after editing `.env`.
4. Or set **`USE_LLM=false`** in `.env` to skip the API entirely.
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

st.subheader("File Upload")
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
        st.dataframe(load_result.table, use_container_width=True)
        with st.spinner("Autonomous agent is analyzing your dataset..."):
            try:
                st.session_state.auto_insights, st.session_state.auto_visuals = (
                    st.session_state.agent.generate_insights_and_charts(st.session_state.tools.df)
                )
            except Exception as exc:
                st.session_state.auto_insights = [
                    "LLM insight step failed. Check `GROQ_API_KEY` in `.env` (valid Groq key starting with `gsk_`). "
                    f"Error: {exc}"
                ]
                st.session_state.auto_visuals = st.session_state.agent.generate_automatic_visualizations()
                st.warning(
                    "Automatic insights used a fallback message because the Groq API returned an error. "
                    "Charts and chat may still work with rule-based planning."
                )
            quality_result = st.session_state.tools.data_quality_report()
            st.session_state.quality_report = quality_result.quality_report or {}
        st.success("Load complete.")

if st.session_state.join_suggestions:
    st.subheader("Suggested Cross-File Joins")
    with st.expander("Review and apply suggested join", expanded=False):
        suggestion_labels = [
            f"{s['left_dataset']} <> {s['right_dataset']} on {s['join_column']} (score={s['overlap_score']})"
            for s in st.session_state.join_suggestions
        ]
        selected_idx = st.selectbox("Join suggestion", options=list(range(len(suggestion_labels))), format_func=lambda i: suggestion_labels[i])
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

if st.session_state.auto_insights:
    st.subheader("Insights")
    for item in st.session_state.auto_insights[:6]:
        st.markdown(f"- {item}")

if st.session_state.auto_visuals:
    st.subheader("Interactive Charts")
    for visual in st.session_state.auto_visuals:
        st.markdown(f"- {visual.text}")
        if visual.plotly_fig is not None:
            st.plotly_chart(visual.plotly_fig, use_container_width=True)

if st.session_state.quality_report:
    st.subheader("Data Quality Report")
    with st.expander("View quality diagnostics", expanded=False):
        st.json(st.session_state.quality_report)

if st.session_state.tools.df is not None:
    st.subheader("Chat with your data")
    user_query = st.chat_input("Ask a question, e.g. 'Clean data and summarize key insights'")

    if user_query:
        st.session_state.chat_log.append({"role": "user", "content": user_query})
        mode = "technical" if explanation_mode == "Technical Explanation" else "simple"
        try:
            response = st.session_state.agent.run(user_query, explanation_mode=mode)
        except Exception as exc:
            st.error(f"Could not process request with the LLM backend. Please verify GROQ_API_KEY and try again. Details: {exc}")
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
                    st.text(response_obj.plan_summary or "Plan not available.")
                if show_reasoning:
                    with st.expander("Agent reasoning steps"):
                        for i, step in enumerate(response_obj.steps, start=1):
                            st.write(f"{i}. Tool: `{step.tool}` | Reason: {step.reason or 'N/A'} | Params: {step.params}")

                for out in response_obj.tool_outputs:
                    if out.table is not None:
                        st.dataframe(out.table, use_container_width=True)
                    if out.plotly_fig is not None:
                        st.plotly_chart(out.plotly_fig, use_container_width=True)

else:
    st.info("Upload and load a CSV file to begin.")
