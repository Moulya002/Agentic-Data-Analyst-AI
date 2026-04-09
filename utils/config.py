import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _clean_secret(value: str) -> str:
    """Strip whitespace and optional surrounding quotes from env secrets."""
    v = (value or "").strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        v = v[1:-1].strip()
    return v


@dataclass
class Settings:
    groq_api_key: str = _clean_secret(os.getenv("GROQ_API_KEY", ""))
    groq_model: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    enable_reasoning_trace: bool = os.getenv("ENABLE_REASONING_TRACE", "true").lower() == "true"
    max_memory_messages: int = int(os.getenv("MAX_MEMORY_MESSAGES", "20"))
    memory_file_path: str = os.getenv("MEMORY_FILE_PATH", ".memory/chat_history.json")


settings = Settings()
