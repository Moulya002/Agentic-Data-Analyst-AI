import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = True) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _clean_secret(value: str) -> str:
    """Strip whitespace and optional surrounding quotes from env secrets."""
    v = (value or "").strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        v = v[1:-1].strip()
    return v


def _is_placeholder_groq_key(key: str) -> bool:
    if not key:
        return True
    k = key.strip().lower()
    if k.startswith("your_") or k.startswith("sk-xxxxx") or "placeholder" in k:
        return True
    if k in ("your_groq_api_key_here", "gsk_your_key_here"):
        return True
    return False


@dataclass
class Settings:
    groq_api_key: str = _clean_secret(os.getenv("GROQ_API_KEY", ""))
    groq_model: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    # Set USE_LLM=false in .env to run 100% offline (no Groq calls — avoids flaky keys).
    use_llm: bool = _env_bool("USE_LLM", default=True)
    enable_reasoning_trace: bool = os.getenv("ENABLE_REASONING_TRACE", "true").lower() == "true"
    max_memory_messages: int = int(os.getenv("MAX_MEMORY_MESSAGES", "20"))
    memory_file_path: str = os.getenv("MEMORY_FILE_PATH", ".memory/chat_history.json")

    @property
    def llm_enabled(self) -> bool:
        return self.use_llm and not _is_placeholder_groq_key(self.groq_api_key)


settings = Settings()
