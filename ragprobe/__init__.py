"""
ragprobe — Adversarial evaluation framework for RAG systems on financial document Q&A.

Probes a RAG pipeline with 5 categories of adversarial prompts, scores responses
with an LLM-as-judge across faithfulness / relevance / context recall, classifies
safety failures, and persists everything to SQLite for audit.

Quick start::

    from ragprobe.probe_engine import run_session

    session_id = run_session(mode="baseline")
    # → results stored in SQLite; render reports with ragprobe.reporter
"""

__version__ = "0.1.0"
