"""
SpectrumLens — Pipeline Bootstrap Script
=========================================
A single entry point to run any stage of the SpectrumLens pipeline.

Usage:
    python run_pipeline.py --ingest          # Stage 1: Parse PDFs → JSON chunks
    python run_pipeline.py --upload          # Stage 2: Embed with BGE-M3 → Supabase
    python run_pipeline.py --demo            # Launch offline demo (no Supabase)
    python run_pipeline.py --app             # Launch full Evidence Panel UI (Supabase)
    python run_pipeline.py --eval            # Run Precision@K evaluation
    python run_pipeline.py --full            # ingest + upload + launch app

Prerequisites:
    • Copy .env.example → .env and fill in API keys
    • Run supabase_schema.sql in Supabase SQL Editor (before --upload)
    • Place clinical PDFs in data/raw_pdfs/
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("SpectrumLens-Runner")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
RAW_PDFS      = ROOT / "data" / "raw_pdfs"
CHUNKS_OUTPUT = ROOT / "data" / "processed_chunks" / "day1_chunks_output.json"
ENV_FILE      = ROOT / ".env"

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _banner(title: str) -> None:
    w = 68
    logger.info("═" * w)
    logger.info(f"  {title}")
    logger.info("═" * w)


def _check_env(*keys: str) -> bool:
    """Return True if all required env keys are set and non-placeholder."""
    ok = True
    for key in keys:
        val = os.environ.get(key, "")
        if not val or "your-project" in val or val.endswith("..."):
            logger.error(f"  ❌ {key} is not configured in .env")
            ok = False
        else:
            logger.info(f"  ✅ {key} configured")
    return ok


def _run(cmd: list[str], **kwargs) -> int:
    """Run a subprocess command and return the exit code."""
    logger.info(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    return result.returncode


# ─── Stage 1: Ingest ──────────────────────────────────────────────────────────
def stage_ingest() -> bool:
    _banner("Stage 1 — Document Ingestion (day1_ingestion.py)")

    pdf_files = list(RAW_PDFS.glob("*.pdf"))
    if not pdf_files:
        logger.error(
            f"No PDF files found in {RAW_PDFS}.\n"
            "  Add your clinical PDFs there and re-run."
        )
        return False

    logger.info(f"Found {len(pdf_files)} PDF(s) in {RAW_PDFS}:")
    for pdf in pdf_files:
        logger.info(f"  • {pdf.name}  ({pdf.stat().st_size // 1024} KB)")

    rc = _run([sys.executable, str(ROOT / "day1_ingestion.py")])
    if rc != 0:
        logger.error(f"Ingestion failed (exit code {rc})")
        return False

    if CHUNKS_OUTPUT.exists():
        size_kb = CHUNKS_OUTPUT.stat().st_size // 1024
        logger.info(f"✅ Chunks saved → {CHUNKS_OUTPUT} ({size_kb} KB)")
    else:
        logger.error("Ingestion completed but chunks file not found.")
        return False

    return True


# ─── Stage 2: Upload ──────────────────────────────────────────────────────────
def stage_upload() -> bool:
    _banner("Stage 2 — Embed & Upload to Supabase (day2_retrieval.py --upload)")

    logger.info("Checking environment variables…")
    if not _check_env("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "GROQ_API_KEY"):
        logger.error(
            "Configure .env before uploading.\n"
            "  Copy .env.example → .env and fill in your keys."
        )
        return False

    if not CHUNKS_OUTPUT.exists():
        logger.error(
            f"Chunks file not found: {CHUNKS_OUTPUT}\n"
            "  Run --ingest first."
        )
        return False

    logger.info("Starting embedding + upload (BGE-M3, ~2.27 GB model on first run)…")
    rc = _run([sys.executable, str(ROOT / "day2_retrieval.py"), "--upload"])
    if rc != 0:
        logger.error(f"Upload failed (exit code {rc})")
        return False

    logger.info("✅ Chunks embedded and uploaded to Supabase.")
    return True


# ─── Launch: Offline Demo ─────────────────────────────────────────────────────
def stage_demo() -> bool:
    _banner("Launching Offline Demo (demo_app.py — no Supabase required)")

    if not CHUNKS_OUTPUT.exists():
        logger.error(
            f"Chunks file not found: {CHUNKS_OUTPUT}\n"
            "  Run --ingest first to generate the local index."
        )
        return False

    logger.info("Launching demo_app.py on http://localhost:8501 …")
    logger.info("  Ctrl+C to stop the server.")
    rc = _run([sys.executable, "-m", "streamlit", "run", str(ROOT / "demo_app.py")])
    return rc == 0


# ─── Launch: Full App ─────────────────────────────────────────────────────────
def stage_app() -> bool:
    _banner("Launching Full Evidence Panel UI (app.py — Supabase required)")

    logger.info("Checking environment variables…")
    if not _check_env("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "GROQ_API_KEY"):
        logger.error(
            "Configure .env before launching the full app.\n"
            "  Or use --demo for the offline version."
        )
        return False

    logger.info("Launching app.py on http://localhost:8501 …")
    logger.info("  Ctrl+C to stop the server.")
    rc = _run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")])
    return rc == 0


# ─── Evaluation ───────────────────────────────────────────────────────────────
def stage_eval(category: str = None, no_gen: bool = False) -> bool:
    _banner("Precision@K Evaluation Harness (evaluate.py)")

    cmd = [sys.executable, str(ROOT / "evaluate.py")]
    if category:
        cmd += ["--category", category]
    if no_gen:
        cmd.append("--no-generation")

    rc = _run(cmd)
    return rc == 0


# ─── Deploy Schema ────────────────────────────────────────────────────────────
def stage_deploy_schema() -> bool:
    _banner("Deploy Schema to Supabase (deploy_schema.py)")
    rc = _run([sys.executable, str(ROOT / "deploy_schema.py")])
    return rc == 0


# ─── Full Pipeline ────────────────────────────────────────────────────────────
def stage_full() -> bool:
    _banner("Full Pipeline: Ingest → Upload → Launch App")

    if not stage_ingest():
        return False
    if not stage_upload():
        return False
    return stage_app()


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SpectrumLens — Pipeline Bootstrap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --ingest           # Parse PDFs → chunks JSON
  python run_pipeline.py --upload           # Embed + upload to Supabase
  python run_pipeline.py --demo             # Offline Streamlit demo
  python run_pipeline.py --app              # Full Streamlit UI (Supabase)
  python run_pipeline.py --eval             # Precision@K evaluation
  python run_pipeline.py --eval --no-gen    # Retrieval-only evaluation (fast)
  python run_pipeline.py --full             # Ingest + upload + launch app
        """,
    )

    parser.add_argument("--ingest",  action="store_true", help="Stage 1: Parse PDFs → JSON chunks")
    parser.add_argument("--upload",  action="store_true", help="Stage 2: Embed with BGE-M3 → Supabase")
    parser.add_argument("--demo",    action="store_true", help="Launch offline demo (no Supabase)")
    parser.add_argument("--app",     action="store_true", help="Launch full Evidence Panel UI (Supabase)")
    parser.add_argument("--eval",    action="store_true", help="Run Precision@K evaluation")
    parser.add_argument("--full",    action="store_true", help="Run full pipeline: ingest + upload + app")
    parser.add_argument("--deploy-schema", action="store_true", help="Deploy database schema to Supabase")
    parser.add_argument("--category", type=str, choices=["factual", "inferential", "oos", "adversarial"],
                        help="Eval category filter (use with --eval)")
    parser.add_argument("--no-gen",  action="store_true",
                        help="Skip LLM generation in eval (retrieval metrics only)")

    args = parser.parse_args()

    if not any([args.ingest, args.upload, args.demo, args.app, args.eval, args.full, args.deploy_schema]):
        parser.print_help()
        sys.exit(0)

    success = True

    if args.full:
        success = stage_full()
    else:
        if args.deploy_schema:
            success = stage_deploy_schema() and success
        if args.ingest:
            success = stage_ingest() and success
        if args.upload:
            success = stage_upload() and success
        if args.demo:
            success = stage_demo() and success
        if args.app:
            success = stage_app() and success
        if args.eval:
            success = stage_eval(category=args.category, no_gen=args.no_gen) and success

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
