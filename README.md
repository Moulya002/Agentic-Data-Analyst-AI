# Agentic Data Analyst AI

Production-style multi-agent data analyst with Groq LLM backend (`llama3-70b-8192`), reflection, interactive Plotly charts, and persistent memory.

## Features

- Upload one or multiple CSV files
- Automatic business insights and visualizations
- Multi-agent workflow:
  - Planner Agent
  - Execution Agent
  - Insight Agent + reflection step
- Data quality report (missing values, duplicates, anomalies)
- Explanation modes (simple / technical)
- Persistent memory on disk
- Cross-file join suggestions with apply-join action

## Project Structure

```bash
Agentic Data Analyst/
├── app.py
├── agent.py
├── planner.py
├── executor.py
├── insight_agent.py
├── tools.py
├── memory.py
├── requirements.txt
├── .env.example
└── utils/
    ├── config.py
    ├── llm_client.py
    ├── logger.py
    └── retry.py
```

## Setup

1) Create and activate virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Configure environment:

```bash
cp .env.example .env
```

4) Edit `.env`:

- `USE_LLM=true` — set to `false` to run **without** Groq (local rules only; no API key needed).
- `GROQ_API_KEY=your_groq_api_key_here` — real key from [console.groq.com](https://console.groq.com/) (`gsk_...`).
- `GROQ_MODEL=llama3-70b-8192`
- `ENABLE_REASONING_TRACE=true`
- `MAX_MEMORY_MESSAGES=20`
- `MEMORY_FILE_PATH=.memory/chat_history.json`

5) Run:

```bash
streamlit run app.py
```

## Notes

- With a valid key, the app uses Groq Chat Completions (`llama3-70b-8192` by default).
- With `USE_LLM=false` or an invalid/placeholder key, the app still runs using **offline** planning and template insights.

Connect with me at:

Mail: moulyarb02@gmail.com

LinkedIn: [Moulya R B](https://www.linkedin.com/in/moulyarb/)
