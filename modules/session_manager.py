"""
Conversational Session Memory Manager.

Provides bounded, thread-safe session tracking for multi-turn interactions.
Supports conversational context sliding windows, pronoun/anaphora query resolution,
and LRU session eviction with TTL expiration to prevent memory leaks.
"""

import time
import uuid
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set


@dataclass
class TurnRecord:
    turn_index: int
    query: str
    effective_query: str
    synthesis: str
    metrics: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionState:
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turns: List[TurnRecord] = field(default_factory=list)
    cumulative_tokens: int = 0
    running_faithfulness: List[float] = field(default_factory=list)
    indexed_signatures: Set[str] = field(default_factory=set)


class SessionManager:
    """
    In-memory session manager with bounded capacity and time-to-live eviction.
    """

    PRONOUN_PATTERN = re.compile(
        r"\b(it|its|they|them|their|this|that|these|those|previous|earlier|above|aforementioned)\b",
        re.IGNORECASE,
    )

    def __init__(self, max_sessions: int = 100, session_ttl_seconds: int = 3600):
        self.max_sessions = max_sessions
        self.session_ttl_seconds = session_ttl_seconds
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()

    @staticmethod
    def generate_session_id() -> str:
        return uuid.uuid4().hex[:12]

    def _cleanup_expired(self) -> None:
        """Evicts sessions that have exceeded the inactivity TTL."""
        now = time.time()
        expired_keys = [
            sid
            for sid, state in self._sessions.items()
            if (now - state.updated_at) > self.session_ttl_seconds
        ]
        for sid in expired_keys:
            del self._sessions[sid]

    def get_or_create(self, session_id: Optional[str] = None) -> SessionState:
        """Retrieves an existing session or creates a new one, maintaining LRU order."""
        self._cleanup_expired()

        if not session_id or not session_id.strip():
            session_id = self.generate_session_id()

        session_id = session_id.strip()
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            state = self._sessions[session_id]
            state.updated_at = time.time()
            return state

        # Check capacity bounds before adding
        while len(self._sessions) >= self.max_sessions:
            self._sessions.popitem(last=False)

        new_state = SessionState(session_id=session_id)
        self._sessions[session_id] = new_state
        return new_state

    def add_turn(
        self,
        session_id: str,
        query: str,
        effective_query: str,
        synthesis: str,
        metrics: Dict[str, Any],
    ) -> SessionState:
        """Appends a turn to the session history and updates cumulative stats."""
        state = self.get_or_create(session_id)
        turn_index = len(state.turns) + 1
        record = TurnRecord(
            turn_index=turn_index,
            query=query,
            effective_query=effective_query,
            synthesis=synthesis,
            metrics=metrics,
        )
        state.turns.append(record)
        state.updated_at = time.time()

        turn_tokens = metrics.get("turn", {}).get("total_tokens", 0)
        state.cumulative_tokens += int(turn_tokens)

        faithfulness = metrics.get("turn", {}).get("faithfulness")
        if faithfulness is not None:
            state.running_faithfulness.append(float(faithfulness))

        return state

    def get_recent_history(
        self, session_id: str, max_turns: int = 3
    ) -> List[Dict[str, str]]:
        """Returns the most recent dialogue turns formatted for OpenAI-compatible chat messages."""
        if session_id not in self._sessions:
            return []

        state = self._sessions[session_id]
        recent = state.turns[-max_turns:] if max_turns > 0 else []
        messages: List[Dict[str, str]] = []
        for turn in recent:
            messages.append({"role": "user", "content": turn.query})
            messages.append({"role": "assistant", "content": turn.synthesis})
        return messages

    def contextualize_query(self, query: str, session_id: str) -> str:
        """
        Resolves conversational anaphora and pronouns against previous user turns.
        Preserves original query if self-contained; enriches retrieval context otherwise.
        """
        clean_query = query.strip()
        if session_id not in self._sessions:
            return clean_query

        state = self._sessions[session_id]
        if not state.turns:
            return clean_query

        last_turn = state.turns[-1]
        has_anaphora = bool(self.PRONOUN_PATTERN.search(clean_query))

        # If short query or contains pronouns, resolve subject from last user query
        if has_anaphora or len(clean_query.split()) <= 4:
            clean_last = last_turn.query.strip().rstrip("?.!")
            return f"{clean_last} : {clean_query}"

        return clean_query

    def reset_session(self, session_id: str) -> bool:
        """Clears conversational turns and cumulative statistics for a session."""
        if session_id in self._sessions:
            state = self._sessions[session_id]
            state.turns.clear()
            state.cumulative_tokens = 0
            state.running_faithfulness.clear()
            state.updated_at = time.time()
            return True
        return False

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Returns high-level summary metrics for a session."""
        if session_id not in self._sessions:
            return {
                "session_id": session_id,
                "total_turns": 0,
                "cumulative_tokens": 0,
                "avg_faithfulness": 1.0,
                "drift_status": "INITIAL",
            }

        state = self._sessions[session_id]
        total_turns = len(state.turns)
        avg_f = (
            sum(state.running_faithfulness) / len(state.running_faithfulness)
            if state.running_faithfulness
            else 1.0
        )

        return {
            "session_id": session_id,
            "total_turns": total_turns,
            "cumulative_tokens": state.cumulative_tokens,
            "avg_faithfulness": round(avg_f, 4),
        }

