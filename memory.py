import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class MemoryStore:
    """
    In-memory conversation storage.
    Keep it simple for now, and can be replaced with Redis/DB later.
    """

    messages: List[Dict[str, str]] = field(default_factory=list)
    max_messages: int = 20
    storage_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.storage_path:
            self._load()

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]
        self._save()

    def recent(self, limit: int = 8) -> List[Dict[str, str]]:
        return self.messages[-limit:]

    def clear(self) -> None:
        self.messages.clear()
        self._save()

    def _load(self) -> None:
        path = Path(self.storage_path)  # type: ignore[arg-type]
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self.messages = [
                    {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
                    for item in data
                    if isinstance(item, dict)
                ][-self.max_messages :]
        except Exception:
            self.messages = []

    def _save(self) -> None:
        if not self.storage_path:
            return
        path = Path(self.storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.messages, indent=2), encoding="utf-8")
