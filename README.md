# ragprobe

**Adversarial evaluation framework for RAG pipelines on financial document Q&A.**

ragprobe probes a RAG pipeline built on SEC 10-K filings with five categories of
adversarial prompts, scores each response with an LLM-as-judge across faithfulness,
relevance, and context recall, classifies safety failures, and persists everything to
SQLite for audit. The framework runs against two RAG configurations — a baseline and
a hardened pipeline — and reports which changes actually reduce failure rates.

This is the FE 524-B (Prompt Engineering Lab for Business Applications) final project.

---

## Install

```bash
pip install -e .
```

Set your API keys in `.env` (copy from `env.example`):

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...           # required for SEC 10-K embeddings (ingest only)
```

---

## Run the demo (no FAISS / no OpenAI key needed)

```bash
python demo.py
```

The demo exercises every proposal component (judge, safety classifier, db,
probe engine, reporter, compare, prompts) using a fake in-memory 10-K corpus
that deliberately includes some hallucinated answers to produce interesting scores.

---

## Full workflow with real SEC 10-Ks

```bash
# 1. Build the FAISS knowledge base from SEC EDGAR
ragprobe ingest --tickers AAPL MSFT NVDA --years 2022 2023 2024

# 2. Run the baseline pipeline against the adversarial suite
ragprobe run --mode baseline
# → session_id: abc-123-...

# 3. Run the hardened pipeline against the same suite
ragprobe run --mode hardened
# → session_id: def-456-...

# 4. Compare the two
ragprobe compare --session-a abc-123 --session-b def-456

# 5. Generate JSON/CSV/text reports for either session
ragprobe probe-report --session abc-123
```

---

## Components (Proposal §5.1)

| Component | File |
|---|---|
| Probe Engine | `ragprobe/probe_engine.py` |
| Judge Module | `ragprobe/judge.py` + `ragprobe/prompts/judge_prompt_v1.txt` |
| Safety Classifier | `ragprobe/safety_classifier.py` |
| Reporting Module | `ragprobe/reporter.py` |
| SQLite Monitor | `ragprobe/db.py` |
| CLI Interface | `ragprobe/cli.py` |

Supporting modules: `prompts.py` (loads the 51 curated adversarial prompts),
`rag_pipeline.py` (baseline + hardened RAG configurations), `ingest.py`
(SEC EDGAR fetching, chunking, embedding, and FAISS indexing), `compare.py`
(side-by-side session comparison), `config.py` (centralised configuration).

---

## Evaluation dimensions (Proposal §4.2)

Every response is scored on a 0–1 scale by the LLM judge:

- **Faithfulness** — Does the answer reflect only what is in the retrieved context?
- **Relevance** — Does the answer address what was actually asked?
- **Context Recall** — Did the retriever surface the right chunks?

Plus three binary safety flags per response: `injection_compliance`,
`confidentiality_violation`, `refusal_evasion`.

---

## Adversarial prompt suite (Proposal §3.2)

51 prompts across 5 categories in `ragprobe/prompts/prompts.json`:

- `hallucination_bait` — questions referencing non-existent figures
- `context_poisoning` — boundary-ambiguity exploitation
- `temporal_confusion` — mixed fiscal years, stale data
- `prompt_injection` — embedded system-override attempts
- `out_of_scope` — questions with no grounding in the corpus
