"""In-memory chat sessions with a short history window and TTL expiry."""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

Message = Dict[str, str]


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    history: List[Message] = field(default_factory=list)
    turns: int = 0
    troubleshooting: Optional[Any] = None   # TroubleshootingState while a diagnosis tree is active

    def append(self, user: str, assistant: str, max_turns: int) -> None:
        self.history.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
        self.history = self.history[-2 * max_turns:] if max_turns > 0 else []
        self.turns += 1
        self.updated_at = time.time()


class SessionStore:
    def __init__(self, ttl_minutes: int = 30, max_turns: int = 1):
        self.ttl = ttl_minutes * 60
        self.max_turns = max_turns
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> Session:
        with self._lock:
            self._expire()
            s = self._sessions.get(session_id)
            if s is None:
                s = self._sessions[session_id] = Session(session_id)
            return s

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            self._expire()
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def record(self, session_id: str, user: str, assistant: str) -> Session:
        s = self.get_or_create(session_id)
        with self._lock:
            s.append(user, assistant, self.max_turns)
        return s

    def __len__(self) -> int:
        return len(self._sessions)

    def _expire(self) -> None:
        cutoff = time.time() - self.ttl
        for sid in [k for k, s in self._sessions.items() if s.updated_at < cutoff]:
            del self._sessions[sid]
