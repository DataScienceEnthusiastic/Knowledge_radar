"""
Unit and integration tests for conversational memory, live turn evaluation,
evidence claim auditing, uncertainty detection, and quality drift tracking.
"""

import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app import app
from modules.session_manager import SessionManager, SessionState
from modules.turn_evaluator import audit_claims_and_evidence, evaluate_turn


@pytest.fixture
def test_client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Session Manager & LRU Memory Tests
# ---------------------------------------------------------------------------

def test_session_creation_and_history():
    mgr = SessionManager(max_sessions=5, session_ttl_seconds=3600)
    session = mgr.get_or_create("session_1")
    assert session.session_id == "session_1"
    assert len(session.turns) == 0

    # Add turn
    metrics = {"turn": {"total_tokens": 120, "faithfulness": 0.95}}
    mgr.add_turn("session_1", "What is RAG?", "What is RAG?", "RAG is retrieval-augmented generation.", metrics)

    history = mgr.get_recent_history("session_1", max_turns=3)
    assert len(history) == 2  # 1 user + 1 assistant message
    assert history[0]["content"] == "What is RAG?"
    assert history[1]["content"] == "RAG is retrieval-augmented generation."

    summary = mgr.get_session_summary("session_1")
    assert summary["total_turns"] == 1
    assert summary["cumulative_tokens"] == 120
    assert summary["avg_faithfulness"] == 0.95


def test_session_lru_capacity_eviction():
    mgr = SessionManager(max_sessions=3, session_ttl_seconds=3600)
    mgr.get_or_create("s1")
    mgr.get_or_create("s2")
    mgr.get_or_create("s3")
    assert len(mgr._sessions) == 3

    # Adding a 4th session should evict the oldest ('s1')
    mgr.get_or_create("s4")
    assert len(mgr._sessions) == 3
    assert "s1" not in mgr._sessions
    assert "s2" in mgr._sessions
    assert "s4" in mgr._sessions


def test_session_ttl_expiration():
    mgr = SessionManager(max_sessions=10, session_ttl_seconds=1)
    state = mgr.get_or_create("expire_me")
    state.updated_at = time.time() - 10.0  # Simulate 10 seconds ago

    # Accessing should evict expired session
    mgr._cleanup_expired()
    assert "expire_me" not in mgr._sessions


def test_query_contextualization_anaphora():
    mgr = SessionManager(max_sessions=5)
    mgr.get_or_create("s_context")
    metrics = {"turn": {"total_tokens": 50, "faithfulness": 1.0}}
    mgr.add_turn("s_context", "What is the revenue of Nvidia in 2024?", "What is the revenue of Nvidia in 2024?", "It was $60.9B.", metrics)

    # Next turn has pronoun 'it'
    resolved = mgr.contextualize_query("What were its profit margins?", "s_context")
    assert "Nvidia in 2024" in resolved
    assert "profit margins" in resolved

    # Self-contained query should remain unchanged
    self_contained = mgr.contextualize_query("Provide an overview of Transformer attention mechanisms in deep learning", "s_context")
    assert self_contained == "Provide an overview of Transformer attention mechanisms in deep learning"


def test_session_reset():
    mgr = SessionManager()
    mgr.get_or_create("s_reset")
    metrics = {"turn": {"total_tokens": 80, "faithfulness": 0.8}}
    mgr.add_turn("s_reset", "Hello", "Hello", "Hi there", metrics)
    assert mgr.get_session_summary("s_reset")["total_turns"] == 1

    cleared = mgr.reset_session("s_reset")
    assert cleared is True
    summary = mgr.get_session_summary("s_reset")
    assert summary["total_turns"] == 0
    assert summary["cumulative_tokens"] == 0


# ---------------------------------------------------------------------------
# 2. Turn Evaluator, Claims Audit & Uncertainty Detection Tests
# ---------------------------------------------------------------------------

def test_audit_claims_grounded_vs_unverified():
    doc_context = [
        {"source": "Report.pdf", "chunk_index": 1, "text": "Acme Corp reported $500M revenue in 2024 with 25% growth."}
    ]
    web_context = []

    # Case 1: Grounded synthesis
    grounded_synthesis = "Acme Corp achieved $500M revenue in 2024. The firm recorded 25% growth."
    claims, unverified = audit_claims_and_evidence(grounded_synthesis, doc_context, web_context)
    assert len(claims) >= 2
    assert len(unverified) == 0
    assert all(c["status"] == "GROUNDED" for c in claims)

    # Case 2: Ungrounded / Hallucinated assertion
    hallucinated_synthesis = "Acme Corp acquired CyberTech in 2026 for $900M."
    claims_h, unverified_h = audit_claims_and_evidence(hallucinated_synthesis, doc_context, web_context)
    assert len(unverified_h) >= 1
    assert any(c["status"] == "LOW_EVIDENCE" for c in claims_h)


def test_audit_claims_explicit_uncertainty():
    doc_context = [{"source": "Doc", "chunk_index": 0, "text": "Operating expenses rose."}]
    web_context = []

    uncertain_synthesis = "Operating expenses rose. [⚠️ Verification Required: exact operating income not specified in document]."
    claims, unverified = audit_claims_and_evidence(uncertain_synthesis, doc_context, web_context)
    assert len(unverified) >= 1
    assert any(c["status"] == "VERIFICATION_REQUIRED" for c in claims)


def test_evaluate_turn_and_drift_detection():
    doc_context = [{"source": "Doc", "chunk_index": 0, "text": "Quantum computing enables rapid factorization using Shor's algorithm."}]
    telemetry = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "synthesis_latency_ms": 320.0}

    # Turn 1: High faithfulness baseline
    turn_1 = evaluate_turn(
        query="How does quantum computing factor numbers?",
        synthesis="Quantum computing uses Shor's algorithm for rapid factorization [Doc].",
        doc_context=doc_context,
        web_context=[],
        telemetry=telemetry,
        retrieval_latency_ms=15.0,
        web_latency_ms=0.0,
        session_history_faithfulness=[]
    )
    assert turn_1["drift"]["status"] == "INITIAL"
    assert turn_1["metrics"]["turn"]["faithfulness"] > 0.70
    assert turn_1["metrics"]["turn"]["citation_validity"] == 1.0

    # Turn 2: Quality drop should trigger DRIFT_WARNING
    turn_2 = evaluate_turn(
        query="What is quantum gravity?",
        synthesis="Quantum gravity predicts entirely unverified extraterrestrial warp travel across the galaxy.",
        doc_context=doc_context,
        web_context=[],
        telemetry=telemetry,
        retrieval_latency_ms=10.0,
        web_latency_ms=0.0,
        session_history_faithfulness=[0.95, 0.92]
    )
    assert turn_2["drift"]["status"] == "DRIFT_WARNING"
    assert "Quality Drift Detected" in turn_2["drift"]["message"]


# ---------------------------------------------------------------------------
# 3. Integration Tests via FastAPI TestClient
# ---------------------------------------------------------------------------

def test_api_audit_multi_turn_flow(test_client):
    with patch("app.generate_response") as mock_gen, \
         patch("app.is_time_sensitive_local", return_value=False), \
         patch("app.vector_store.search_chunks", return_value=[{"source": "Doc1", "chunk_index": 0, "text": "Revenue was $10M."}]):

        mock_gen.return_value = (
            "The revenue was $10M according to [Doc1].",
            "Powered by Cloud LLM",
            {"prompt_tokens": 80, "completion_tokens": 25, "total_tokens": 105, "synthesis_latency_ms": 120.0}
        )

        # Turn 1
        res1 = test_client.post("/api/audit", data={"query": "What is the revenue?", "session_id": "test_session_flow"})
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["session_id"] == "test_session_flow"
        assert data1["turn_index"] == 1
        assert "metrics" in data1
        assert "drift" in data1
        assert "claims_audit" in data1
        assert data1["metrics"]["total_tokens"] == 105
        assert data1["session_summary"]["total_turns"] == 1

        # Turn 2 (with anaphora)
        res2 = test_client.post("/api/audit", data={"query": "Was it higher than last year?", "session_id": "test_session_flow"})
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["turn_index"] == 2
        assert data2["session_summary"]["total_turns"] == 2
        assert "revenue" in data2["effective_query"].lower()


def test_api_session_reset(test_client):
    # Reset session endpoint
    res = test_client.post("/api/session/reset", data={"session_id": "test_session_flow"})
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == "test_session_flow"
    assert data["cleared"] is True
    assert data["summary"]["total_turns"] == 0


def test_gemini_config_and_router_labeling():
    from config import GEMINI_BASE_URL, GEMINI_MODELS
    from modules.llm_router import generate_response

    assert "generativelanguage.googleapis.com" in GEMINI_BASE_URL
    assert "low" in GEMINI_MODELS
    assert "medium" in GEMINI_MODELS
    assert "high" in GEMINI_MODELS
    assert "flash" in GEMINI_MODELS["medium"]

    # Test router badge with Gemini
    with patch("modules.llm_router.OpenAI") as mock_openai_cls:
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance
        mock_choice = MagicMock()
        mock_choice.message.content = "Gemini response synthesis."
        mock_usage = MagicMock(prompt_tokens=40, completion_tokens=20, total_tokens=60)
        mock_instance.chat.completions.create.return_value = MagicMock(choices=[mock_choice], usage=mock_usage)

        output, badge, telemetry = generate_response(
            query="Tell me about Gemini",
            doc_context=[],
            web_context=[],
            is_time_sensitive=False,
            cloud_api_key="AIzaSy_fake_test_key",
            cloud_base_url=GEMINI_BASE_URL,
            cloud_model=GEMINI_MODELS["medium"]
        )

        assert "Gemini" in badge
        assert "gemini-2.0-flash" in badge
        assert telemetry["total_tokens"] == 60
        assert "Gemini response synthesis." in output


def test_multi_source_balanced_retrieval():
    from modules.vector_store import KnowledgeVectorStore
    store = KnowledgeVectorStore()

    # Add 10 chunks from YouTube and 5 chunks from PDF
    yt_chunks = [{"text": f"YouTube chunk {i} discussing neural network training parameters and backpropagation", "source": "YouTube (3Blue1Brown)", "chunk_index": i} for i in range(10)]
    pdf_chunks = [{"text": f"PDF section {i} discussing deep learning attention mechanisms and transformers", "source": "PDF: paper.pdf", "chunk_index": i} for i in range(5)]
    store.add_chunks(yt_chunks)
    store.add_chunks(pdf_chunks)

    summary = store.get_sources_summary()
    assert summary["YouTube (3Blue1Brown)"] == 10
    assert summary["PDF: paper.pdf"] == 5

    # Query with generic cross-source question
    query = "What is the relation between pdf and yt video?"
    results = store.search_chunks(query, top_k=5)

    # Must contain chunks from BOTH sources!
    sources_in_results = {r["source"] for r in results}
    assert "YouTube (3Blue1Brown)" in sources_in_results
    assert "PDF: paper.pdf" in sources_in_results
    assert len(results) >= 5

