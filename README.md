# RagProbe

LLM evaluation and adversarial testing framework for benchmarking hallucination, prompt injection, retrieval quality, faithfulness, relevance, and robustness in Retrieval-Augmented Generation (RAG) systems.

## Overview

RagProbe is a modular framework designed to evaluate the reliability and security of RAG pipelines under adversarial and real-world conditions. The system benchmarks baseline versus hardened retrieval pipelines using automated evaluation workflows, LLM-based judges, and retrieval quality metrics.

The framework supports:

* Hallucination detection
* Prompt injection testing
* Faithfulness and relevance scoring
* Context recall evaluation
* Baseline vs hardened RAG comparison
* Automated experiment reporting and dashboards

Built to simulate production-grade LLM evaluation workflows for modern AI systems and agentic applications.

---

# Features

* Adversarial prompt injection testing
* Hallucination and grounding evaluation
* Automated RAG benchmarking workflows
* FAISS-based vector retrieval
* Baseline vs hardened pipeline comparison
* LLM-as-a-Judge evaluation pipeline
* SEC filing ingestion and chunking
* Experiment dashboards and reporting

---

# Architecture

```text
SEC Filings / Documents
            │
            ▼
      Chunking Pipeline
            │
            ▼
      FAISS Vector Store
            │
            ▼
      Retrieval Pipeline
            │
            ▼
        LLM Response
            │
            ▼
      Evaluation Engine
   ├── Faithfulness
   ├── Relevance
   ├── Context Recall
   ├── Hallucination
   └── Prompt Injection
            │
            ▼
    Dashboard + Reports
```

---

# Tech Stack

## AI / LLM

* OpenAI API
* Retrieval-Augmented Generation (RAG)
* LLM-as-a-Judge Evaluation
* Prompt Engineering

## Backend and Infrastructure

* Python
* FastAPI
* FAISS
* SQLite
* Docker

## Visualization and Reporting

* Plotly Dash
* Matplotlib
* Automated HTML Reports

---

# Project Structure

```text
ragprobe/
│
├── ragprobe/
│   ├── ingest.py
│   ├── rag_pipeline.py
│   ├── probe_engine.py
│   ├── judge.py
│   ├── reporter.py
│   └── compare.py
│
├── tests/
├── dashboard.py
├── run_probe.py
└── README.md
```

---

# Example Evaluation Categories

| Category         | Purpose                                |
| ---------------- | -------------------------------------- |
| Hallucination    | Detect unsupported model responses     |
| Prompt Injection | Evaluate adversarial robustness        |
| Faithfulness     | Measure grounding to retrieved context |
| Relevance        | Evaluate answer quality                |
| Context Recall   | Measure retrieval completeness         |

---

# Running the Project

## Clone Repository

```bash
git clone https://github.com/Venkyyy98/RagProbe.git
cd RagProbe
```

## Install Dependencies

```bash
pip install -e .
```

## Configure Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

## Run Ingestion Pipeline

```bash
ragprobe ingest
```

## Execute Evaluation Pipeline

```bash
ragprobe run --mode baseline
```

## Compare Hardened vs Baseline

```bash
ragprobe compare
```

## Launch Dashboard

```bash
python dashboard.py
```

---

# Future Improvements

* Multi-model benchmarking
* Agentic workflow evaluation
* Real-time attack simulation
* Vector database integrations
* CI/CD evaluation automation
* Human feedback integration

---

# Author

Venkatesh Mudaliar

Master’s in Data Science — Stevens Institute of Technology

Focused on LLM evaluation systems, agentic AI workflows, retrieval pipelines, and AI reliability engineering.

