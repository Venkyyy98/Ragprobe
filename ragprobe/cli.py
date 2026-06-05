"""
cli.py — Command-line interface for ragprobe.

Six sub-commands aligned with the project proposal:

  ragprobe ingest       --tickers AAPL MSFT ...   # SEC 10-K ingestion to FAISS
  ragprobe run          --mode baseline|hardened  # Run the adversarial probe suite
  ragprobe probe-report --session SESSION_ID      # JSON/CSV/text reports for a run
  ragprobe compare      --session-a A --session-b B  # Baseline-vs-hardened analysis
  ragprobe db-summary                              # List all sessions in SQLite
  ragprobe report       --input results.json --format json|html  # Render saved results

Entry point: main()
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Argparse with exit-code 1 on usage errors ─────────────────────────────────

class _Parser(argparse.ArgumentParser):
    """ArgumentParser that exits with code 1 (not 2) on usage errors."""

    def error(self, message):  # type: ignore[override]
        self.print_usage(sys.stderr)
        print(f"error: {message}", file=sys.stderr)
        sys.exit(1)


def _die(msg, code=1):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def _ok(msg):
    print(f"\u2713 {msg}")


# ── Sub-command: report ───────────────────────────────────────────────────────

def _cmd_report(args):
    """Render a results JSON file as JSON or HTML."""
    input_path = Path(args.input)
    if not input_path.is_file():
        _die(f"--input file not found: {input_path}")

    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _die(f"invalid JSON in {input_path}: {exc}")

    if args.format == "json":
        rendered = json.dumps(data, indent=2)
    else:
        rendered = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>ragprobe report</title>"
            "<style>body{font-family:system-ui;margin:2rem;}"
            "pre{background:#f5f5f5;padding:1rem;border-radius:6px;"
            "overflow-x:auto;}</style></head><body>"
            "<h1>ragprobe report</h1>"
            f"<pre>{json.dumps(data, indent=2)}</pre>"
            "</body></html>"
        )

    if args.output:
        out = Path(args.output)
        out.write_text(rendered, encoding="utf-8")
        _ok(f"wrote {out}")
    else:
        print(rendered)


# ── Sub-command: ingest ───────────────────────────────────────────────────────

def _cmd_ingest(args):
    from ragprobe.ingest import build_index
    from ragprobe.config import FAISS_PATH

    index_dir = Path(args.index_dir) if args.index_dir else FAISS_PATH
    tickers   = [t.upper() for t in args.tickers]

    print(f"Ingesting tickers={tickers} years={args.years} \u2192 {index_dir}")
    result = build_index(tickers=tickers, years=args.years, index_dir=index_dir)
    _ok(
        f"ingest complete \u2014 {result['total_chunks']} chunks indexed "
        f"\u2192 {result['index_dir']}"
    )


# ── Sub-command: run (the main proposal command) ──────────────────────────────

def _cmd_run(args):
    from ragprobe.probe_engine import run_session
    from ragprobe.config import FAISS_PATH

    faiss_path = Path(args.faiss_path) if args.faiss_path else FAISS_PATH

    print(
        f"Starting probe run: mode={args.mode}"
        + (f" category={args.category}" if args.category else "")
        + (f" limit={args.limit}" if args.limit else "")
        + (f" faiss={faiss_path}" if args.faiss_path else "")
    )

    session_id = run_session(
        mode            = args.mode,
        category        = args.category,
        limit           = args.limit,
        use_llm_safety  = not args.no_llm_safety,
        faiss_path      = faiss_path,
    )

    _ok(f"probe run complete \u2014 session_id={session_id}")
    print(f"  \u2192 run 'ragprobe probe-report --session {session_id}' to generate reports")


# ── Sub-command: probe-report ─────────────────────────────────────────────────

def _cmd_probe_report(args):
    from ragprobe.reporter import generate_probe_reports

    print(f"Generating reports for session: {args.session}")
    paths = generate_probe_reports(args.session)
    _ok("reports written:")
    for fmt, path in paths.items():
        print(f"  [{fmt:4s}] {path}")


# ── Sub-command: compare ──────────────────────────────────────────────────────

def _cmd_compare(args):
    from ragprobe.compare import compare_sessions
    compare_sessions(args.session_a, args.session_b)


# ── Sub-command: db-summary ───────────────────────────────────────────────────

def _cmd_db_summary(args):  # noqa: ARG001
    from ragprobe.db import init_db, list_sessions
    from ragprobe.config import DB_PATH

    init_db()
    sessions = list_sessions()

    if not sessions:
        print("No sessions found in the database.")
        return

    print(f"\n{'Session ID':<40}  {'Timestamp':<24}  {'Mode':<10}  "
          f"{'Probes':>6}  {'Faith':>6}  {'Rel':>6}  {'Ctx':>6}")
    print("-" * 110)
    for s in sessions:
        faith = f"{s['mean_faithfulness']:.3f}"   if s["mean_faithfulness"]   is not None else "  N/A"
        rel   = f"{s['mean_relevance']:.3f}"      if s["mean_relevance"]      is not None else "  N/A"
        ctx   = f"{s['mean_context_recall']:.3f}" if s["mean_context_recall"] is not None else "  N/A"
        print(
            f"{s['session_id']:<40}  {s['run_timestamp']:<24}  "
            f"{s['pipeline_mode']:<10}  {s['total_probes']:>6}  "
            f"{faith:>6}  {rel:>6}  {ctx:>6}"
        )
    print(f"\n{len(sessions)} session(s) in {DB_PATH}\n")


# ── Parser construction ───────────────────────────────────────────────────────

def _build_parser():
    parser = _Parser(
        prog="ragprobe",
        description="Adversarial evaluation framework for RAG pipelines on financial documents.",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG logging.")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    p_report = sub.add_parser("report",
        help="Render a saved results JSON file as JSON or HTML.")
    p_report.add_argument("--input", required=True, metavar="PATH",
                          help="Path to results JSON file.")
    p_report.add_argument("--format", choices=["json", "html"], default="json",
                          help="Output format (default: json).")
    p_report.add_argument("--output", metavar="PATH",
                          help="Write rendered report to this file (default: stdout).")

    p_ingest = sub.add_parser("ingest",
        help="Fetch, parse, chunk, embed, and index 10-K filings from SEC EDGAR.",
        description=("Download 10-K filings for the given tickers, chunk them at "
                     "~512 tokens, embed with text-embedding-3-small, and build a "
                     "FAISS flat L2 index. Requires OPENAI_API_KEY."))
    p_ingest.add_argument("--tickers", nargs="+", required=True, metavar="TICKER",
                          help="Ticker symbols to ingest (e.g. AAPL MSFT AMZN).")
    p_ingest.add_argument("--years", nargs="+", type=int,
                          default=[2022, 2023, 2024], metavar="YEAR",
                          help="Fiscal years to include (default: 2022 2023 2024).")
    p_ingest.add_argument("--index-dir", dest="index_dir", metavar="DIR", default=None,
                          help="Output directory for FAISS index (default: faiss_index/).")

    p_run = sub.add_parser("run",
        help="Run the full adversarial probe suite against the RAG pipeline.",
        description=("Load prompts.json, fire each prompt at the RAG pipeline, score "
                     "with the LLM judge and safety classifier, and persist all results "
                     "to SQLite. Returns the session ID on completion."))
    p_run.add_argument("--mode", choices=["baseline", "hardened"], default="baseline",
                       help="Pipeline mode (default: baseline).")
    p_run.add_argument("--category", metavar="CAT", default=None,
                       help=("Run only this prompt category: hallucination_bait, "
                             "context_poisoning, temporal_confusion, prompt_injection, "
                             "out_of_scope."))
    p_run.add_argument("--limit", type=int, metavar="N", default=None,
                       help="Stop after N probes (useful for smoke testing).")
    p_run.add_argument("--no-llm-safety", dest="no_llm_safety", action="store_true",
                       help="Disable LLM fallback in safety classifier (keyword-only).")
    p_run.add_argument("--faiss-path", dest="faiss_path", metavar="DIR", default=None,
                       help="FAISS index directory to use (default: faiss_index/).")

    p_probe_report = sub.add_parser("probe-report",
        help="Generate JSON/CSV/text reports for a probe session.")
    p_probe_report.add_argument("--session", required=True, metavar="SESSION_ID",
                                help="Session ID returned by 'ragprobe run'.")

    p_compare = sub.add_parser("compare",
        help="Compare two probe sessions side by side (baseline vs hardened).")
    p_compare.add_argument("--session-a", dest="session_a", required=True,
                           metavar="SESSION_ID", help="Reference session ID.")
    p_compare.add_argument("--session-b", dest="session_b", required=True,
                           metavar="SESSION_ID", help="Comparison session ID.")

    sub.add_parser("db-summary",
        help="Print a summary of all sessions in the SQLite store.")

    return parser


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = _build_parser()
    args   = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        if args.command == "report":
            _cmd_report(args)
        elif args.command == "ingest":
            _cmd_ingest(args)
        elif args.command == "run":
            _cmd_run(args)
        elif args.command == "probe-report":
            _cmd_probe_report(args)
        elif args.command == "compare":
            _cmd_compare(args)
        elif args.command == "db-summary":
            _cmd_db_summary(args)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        logger.debug("Unhandled exception", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
