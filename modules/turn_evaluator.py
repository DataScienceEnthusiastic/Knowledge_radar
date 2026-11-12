"""
Live Turn Evaluator, Claim Grounding, and Drift Detection Engine.

Provides real-time evaluation of LLM generation against retrieved multi-modal
context: faithfulness scoring, citation integrity validation, claim-level
evidence alignment, uncertainty detection, and session-wide quality drift tracking.
"""

import re
from typing import List, Dict, Any, Optional, Tuple

STOPWORDS = {
    "the", "a", "an", "is", "was", "are", "were", "in", "on", "of", "and", "or",
    "for", "to", "with", "by", "at", "from", "as", "that", "this", "it", "its",
    "be", "been", "have", "has", "had", "do", "does", "did", "but", "not", "we",
    "they", "he", "she", "you", "i", "our", "their", "my", "your", "can", "will"
}

FACTUAL_TRIGGER = re.compile(
    r"(\b\d+(\.\d+)?%?\b|\$\d+|\b(19\d\d|20\d\d)\b|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
)

CITATION_PATTERN = re.compile(
    r"\[(PDF:[^\]]+|Image:[^\]]+|YouTube[^\]]*|WebURL[^\]]*|Doc\s*\d*|Page\s*\d+|Chunk\s*\d+|Source:[^\]]+|\d+)\]",
    re.IGNORECASE
)

UNCERTAINTY_EXPLICIT_PATTERN = re.compile(
    r"\[⚠️[^\]]+\]|\b(unverified|unconfirmed|requires verification|not stated|not specified in document)\b",
    re.IGNORECASE
)


def _tokenize(text: str) -> set:
    """Extracts alphanumeric lowercase content tokens excluding generic stopwords."""
    tokens = set(re.findall(r"\w+", text.lower()))
    return tokens - STOPWORDS


def audit_claims_and_evidence(
    synthesis: str,
    doc_context: List[Dict[str, Any]],
    web_context: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parses synthesis into claims and verifies evidence grounding against retrieved snippets.

    Returns:
        tuple: (all_claims_audit, unverified_claims_list)
    """
    if not synthesis or not synthesis.strip():
        return [], []

    # Combine context content tokens
    context_corpus = " ".join([c.get("text", "") for c in doc_context])
    if web_context:
        context_corpus += " " + " ".join([w.get("snippet", "") + " " + w.get("title", "") for w in web_context])

    context_tokens = _tokenize(context_corpus)

    # Split into sentences
    raw_sentences = re.split(r"(?<=[.!?])\s+", synthesis.strip())
    claims_audit: List[Dict[str, Any]] = []
    unverified_claims: List[Dict[str, Any]] = []

    for idx, sentence in enumerate(raw_sentences):
        clean_sentence = sentence.strip()
        if not clean_sentence or len(clean_sentence) < 10:
            continue

        sentence_tokens = _tokenize(clean_sentence)
        if not sentence_tokens:
            continue

        has_factual = bool(FACTUAL_TRIGGER.search(clean_sentence))
        has_explicit_uncertainty = bool(UNCERTAINTY_EXPLICIT_PATTERN.search(clean_sentence))

        overlap_tokens = sentence_tokens.intersection(context_tokens)
        overlap_ratio = len(overlap_tokens) / len(sentence_tokens)

        # Check factual entity overlap if numbers, dates or capitalized entities exist
        factual_tokens_in_sentence = {m for m in sentence_tokens if bool(re.search(r"\d", m)) or len(m) > 7}
        factual_overlap_ratio = (
            len(factual_tokens_in_sentence.intersection(context_tokens)) / len(factual_tokens_in_sentence)
            if factual_tokens_in_sentence
            else overlap_ratio
        )

        # Grounding classification
        if has_explicit_uncertainty:
            status = "VERIFICATION_REQUIRED"
            confidence = 0.50
            supporting_source = "Explicitly flagged by model as uncertain"
            unverified_claims.append({
                "claim_index": idx + 1,
                "text": clean_sentence,
                "reason": "Model explicitly indicated uncertainty or absence in source."
            })
        elif has_factual and (overlap_ratio < 0.40 or factual_overlap_ratio < 0.50):
            status = "LOW_EVIDENCE"
            confidence = round(min(overlap_ratio, factual_overlap_ratio), 2)
            supporting_source = "No direct matching passage found in retrieved context"
            unverified_claims.append({
                "claim_index": idx + 1,
                "text": clean_sentence,
                "reason": f"Factual assertion with low context evidence (factual overlap: {int(factual_overlap_ratio * 100)}%)."
            })
        elif overlap_ratio >= 0.50:
            status = "GROUNDED"
            confidence = round(overlap_ratio, 2)
            supporting_source = "Strongly supported by multi-modal/web context"
        else:
            status = "PARTIAL_SUPPORT"
            confidence = round(overlap_ratio, 2)
            supporting_source = "General synthesis or moderate context overlap"

        claims_audit.append({
            "claim_index": idx + 1,
            "text": clean_sentence,
            "is_factual": has_factual,
            "status": status,
            "confidence": confidence,
            "overlap_ratio": round(overlap_ratio, 4),
            "supporting_source": supporting_source
        })

    return claims_audit, unverified_claims


def evaluate_turn(
    query: str,
    synthesis: str,
    doc_context: List[Dict[str, Any]],
    web_context: List[Dict[str, Any]],
    telemetry: Dict[str, Any],
    retrieval_latency_ms: float = 0.0,
    web_latency_ms: float = 0.0,
    session_history_faithfulness: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive live evaluation metrics for a single conversational turn,
    including token usage, latency distribution, faithfulness, citation validity,
    claim grounding audit, and cross-turn drift detection.
    """
    claims_audit, unverified_claims = audit_claims_and_evidence(synthesis, doc_context, web_context)

    # 1. Faithfulness (RAG triad grounding)
    synthesis_tokens = _tokenize(synthesis)
    context_corpus = " ".join([c.get("text", "") for c in doc_context])
    if web_context:
        context_corpus += " " + " ".join([w.get("snippet", "") for w in web_context])
    context_tokens = _tokenize(context_corpus)

    if synthesis_tokens and context_tokens:
        supported_tokens = synthesis_tokens.intersection(context_tokens)
        faithfulness = round(len(supported_tokens) / len(synthesis_tokens), 4)
    elif not synthesis_tokens:
        faithfulness = 0.0
    else:
        faithfulness = 1.0 if not doc_context and not web_context else 0.0

    # 2. Answer Relevance (query alignment)
    query_tokens = _tokenize(query)
    if query_tokens and synthesis_tokens:
        q_overlap = len(synthesis_tokens.intersection(query_tokens)) / len(query_tokens)
        answer_relevance = round(min(1.0, 0.4 * q_overlap + 0.6 * (1.0 if faithfulness > 0.4 else 0.5)), 4)
    else:
        answer_relevance = 1.0 if synthesis_tokens else 0.0

    # 3. Unsupported Claim Rate per 100 words
    total_words = max(len(synthesis.split()), 1)
    unsupported_count = len(unverified_claims)
    unsupported_claim_rate = round((unsupported_count / total_words) * 100.0, 2)

    # 4. Citation Integrity
    found_citations = CITATION_PATTERN.findall(synthesis)
    citation_count = len(found_citations)
    citation_validity = 1.0 if citation_count > 0 else 0.0

    # 5. Latency Breakdown
    synthesis_latency_ms = telemetry.get("synthesis_latency_ms", 0.0)
    total_latency_ms = round(retrieval_latency_ms + web_latency_ms + synthesis_latency_ms, 2)

    # 6. Token Usage
    prompt_tokens = telemetry.get("prompt_tokens", 0)
    completion_tokens = telemetry.get("completion_tokens", 0)
    total_tokens = telemetry.get("total_tokens", prompt_tokens + completion_tokens)

    # 7. Quality Consistency & Drift Detection
    history = session_history_faithfulness or []
    if not history:
        drift_status = "INITIAL"
        drift_delta = 0.0
        drift_message = "Turn #1 baseline established."
    else:
        avg_prior = sum(history) / len(history)
        drift_delta = round(faithfulness - avg_prior, 4)

        if faithfulness < (avg_prior - 0.20) or faithfulness < 0.55:
            drift_status = "DRIFT_WARNING"
            pct_drop = abs(int(drift_delta * 100))
            drift_message = f"⚠️ Quality Drift Detected: Faithfulness dropped {pct_drop}% below session average."
        elif faithfulness >= (avg_prior - 0.15) and faithfulness >= 0.70:
            drift_status = "STABLE"
            drift_message = f"🟢 Quality Consistent: Within {abs(int(drift_delta * 100))}% of session average."
        else:
            drift_status = "MODERATE_VARIANCE"
            drift_message = "Quality shows moderate variance against prior turns."

    return {
        "metrics": {
            "turn": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latencies_ms": {
                    "retrieval": round(retrieval_latency_ms, 2),
                    "web": round(web_latency_ms, 2),
                    "synthesis": round(synthesis_latency_ms, 2),
                    "total": total_latency_ms
                },
                "faithfulness": faithfulness,
                "answer_relevance": answer_relevance,
                "unsupported_claim_rate": unsupported_claim_rate,
                "citation_validity": citation_validity,
                "citations_found": citation_count,
                "unverified_claims_count": unsupported_count
            }
        },
        "drift": {
            "status": drift_status,
            "delta": drift_delta,
            "message": drift_message
        },
        "claims_audit": {
            "total_claims": len(claims_audit),
            "unverified_count": len(unverified_claims),
            "claims": claims_audit,
            "unverified_claims": unverified_claims
        }
    }
