"""
Quantum Solutions — LiveKit Voice AI Agent
==========================================
A voice-first Q&A agent that can read and answer questions about:
  - PDF files (digital text-layer + scanned/OCR fallback)
  - Markdown (.md) specs and docs
  - SQL schema dumps, RLS policies, RPCs
  - Plain text, logs, JSON, CSV
  - Long pasted content / Claude AI session transcripts

Architecture
------------
  LiveKit Room  ──►  STT (Deepgram)  ──►  LLM (Claude claude-sonnet-4-6)  ──►  TTS (ElevenLabs / Cartesia)
                                          ▲
                               Document context injected
                               into system prompt at startup

Usage
-----
  # Install dependencies
  pip install "livekit-agents[deepgram,anthropic,elevenlabs,silero]~=1.0"
  pip install pypdf pdfplumber pytesseract pillow

  # Set environment variables (see .env.example below)
  export LIVEKIT_URL=wss://your-project.livekit.cloud
  export LIVEKIT_API_KEY=APIxxxxxxxxxx
  export LIVEKIT_API_SECRET=xxxxxxxxxxxx
  export ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxx
  export DEEPGRAM_API_KEY=xxxxxxxxxxxxxxxx
  export ELEVENLABS_API_KEY=xxxxxxxxxxxxxxxx   # optional — falls back to Cartesia
  export CARTESIA_API_KEY=xxxxxxxxxxxxxxxx     # optional — fallback TTS

  # Run without a document (general Q&A mode)
  python agent.py start

  # Run with a document pre-loaded (voice Q&A over that doc)
  python agent.py start --doc /path/to/file.pdf
  python agent.py start --doc /path/to/schema.sql
  python agent.py start --doc /path/to/README.md

  # Run in dev mode (auto-creates a room)
  python agent.py dev --doc /path/to/report.pdf
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import textwrap
from pathlib import Path

# ── LiveKit Agents ────────────────────────────────────────────────────────────
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import anthropic, deepgram, silero

# ── Optional TTS plugins (pick one) ──────────────────────────────────────────
try:
    from livekit.plugins import elevenlabs as tts_plugin
    TTS_PROVIDER = "elevenlabs"
except ImportError:
    try:
        from livekit.plugins import cartesia as tts_plugin
        TTS_PROVIDER = "cartesia"
    except ImportError:
        tts_plugin = None
        TTS_PROVIDER = "none"

# ── Document reading libraries ────────────────────────────────────────────────
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

logger = logging.getLogger("quantum-agent")

# =============================================================================
# DOCUMENT READER
# =============================================================================

class DocumentReader:
    """
    Reads PDF, MD, SQL, TXT, JSON, CSV and returns clean text
    suitable for injection into the LLM system prompt.

    Strategy per file type:
      PDF  → pdfplumber (text layer) → pypdf fallback → OCR fallback
      text → direct read (utf-8 with latin-1 fallback)
      *    → best-effort text decode
    """

    # Max characters to inject into the system prompt.
    # At ~4 chars/token this is ~20K tokens — leaves headroom for conversation.
    MAX_CHARS = 80_000

    # Extensions treated as plain text
    TEXT_EXTENSIONS = {
        ".md", ".sql", ".txt", ".log", ".json",
        ".csv", ".tsv", ".js", ".ts", ".py",
        ".jsx", ".tsx", ".yaml", ".yml", ".env",
        ".xml", ".html", ".htm", ".sh", ".bash",
    }

    def read(self, path: str | Path) -> tuple[str, str]:
        """
        Returns (file_label, extracted_text).
        file_label is a short human-readable descriptor e.g. "PDF (12 pages)".
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        ext = p.suffix.lower()
        size_kb = p.stat().st_size / 1024

        logger.info(f"Reading document: {p.name} ({size_kb:.1f} KB, type={ext})")

        if ext == ".pdf":
            return self._read_pdf(p)
        elif ext in self.TEXT_EXTENSIONS:
            return self._read_text(p, ext)
        else:
            # Try as text, warn if it looks binary
            try:
                return self._read_text(p, ext)
            except Exception:
                raise ValueError(
                    f"Unsupported file type: {ext}. "
                    "Supported: PDF, MD, SQL, TXT, LOG, JSON, CSV, JS, TS, PY, YAML, XML, HTML"
                )

    # ── PDF ──────────────────────────────────────────────────────────────────

    def _read_pdf(self, p: Path) -> tuple[str, str]:
        """
        Three-tier PDF reading:
        1. pdfplumber — best layout-aware text extraction
        2. pypdf      — fallback if pdfplumber not installed
        3. pytesseract OCR — fallback for scanned/raster PDFs
        """
        pages_text: list[str] = []
        page_count = 0

        # ── Tier 1: pdfplumber ───────────────────────────────────────────────
        if HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(p) as pdf:
                    page_count = len(pdf.pages)
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        # Also try to extract tables if present
                        tables = page.extract_tables() or []
                        table_text = ""
                        for tbl in tables:
                            rows = []
                            for row in tbl:
                                clean_row = [cell or "" for cell in row]
                                rows.append(" | ".join(clean_row))
                            table_text += "\n" + "\n".join(rows)
                        combined = (text + table_text).strip()
                        if combined:
                            pages_text.append(f"[Page {i+1}]\n{combined}")

                if pages_text:
                    full_text = "\n\n".join(pages_text)
                    return (
                        f"PDF ({page_count} pages, text-layer extracted)",
                        self._truncate(full_text),
                    )
                # No text extracted → fall through to OCR
                logger.warning("pdfplumber found no text layer — trying OCR")
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e} — falling back to pypdf")

        # ── Tier 2: pypdf ────────────────────────────────────────────────────
        if HAS_PYPDF:
            try:
                reader = PdfReader(str(p))
                page_count = len(reader.pages)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages_text.append(f"[Page {i+1}]\n{text.strip()}")

                if pages_text:
                    full_text = "\n\n".join(pages_text)
                    return (
                        f"PDF ({page_count} pages, pypdf extraction)",
                        self._truncate(full_text),
                    )
                logger.warning("pypdf found no text layer — trying OCR")
            except Exception as e:
                logger.warning(f"pypdf failed: {e} — falling back to OCR")

        # ── Tier 3: OCR via pytesseract ──────────────────────────────────────
        if HAS_OCR:
            return self._read_pdf_ocr(p)

        raise RuntimeError(
            "Cannot extract PDF text. Install pdfplumber: pip install pdfplumber\n"
            "For scanned PDFs also install: pip install pytesseract pillow"
        )

    def _read_pdf_ocr(self, p: Path) -> tuple[str, str]:
        """OCR fallback using pdftoppm + pytesseract for scanned PDFs."""
        import subprocess
        import tempfile
        import glob

        logger.info("Running OCR on scanned PDF — this may take a moment...")
        pages_text = []

        with tempfile.TemporaryDirectory() as tmpdir:
            # Rasterize at 150 DPI (balance between quality and speed)
            result = subprocess.run(
                ["pdftoppm", "-jpeg", "-r", "150", str(p), f"{tmpdir}/page"],
                capture_output=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "pdftoppm failed. Install poppler-utils:\n"
                    "  Ubuntu/Debian: sudo apt-get install poppler-utils\n"
                    "  macOS:         brew install poppler"
                )

            image_files = sorted(glob.glob(f"{tmpdir}/page-*.jpg"))
            for i, img_path in enumerate(image_files):
                img = Image.open(img_path)
                text = pytesseract.image_to_string(img)
                if text.strip():
                    pages_text.append(f"[Page {i+1} — OCR]\n{text.strip()}")

        if not pages_text:
            raise RuntimeError("OCR produced no text. The PDF may be blank or corrupted.")

        page_count = len(pages_text)
        full_text = "\n\n".join(pages_text)
        return (
            f"PDF ({page_count} pages, OCR extracted — scanned document)",
            self._truncate(full_text),
        )

    # ── Plain text types ─────────────────────────────────────────────────────

    def _read_text(self, p: Path, ext: str) -> tuple[str, str]:
        """Read any text-based file with utf-8 + latin-1 fallback."""
        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = p.read_text(encoding="latin-1")

        ext_labels = {
            ".md": "Markdown document",
            ".sql": "SQL schema / script",
            ".json": "JSON data",
            ".csv": "CSV data",
            ".tsv": "TSV data",
            ".log": "Log file",
            ".py": "Python source",
            ".js": ".js source",
            ".ts": "TypeScript source",
            ".yaml": "YAML config",
            ".yml": "YAML config",
            ".xml": "XML document",
            ".html": "HTML document",
            ".sh": "Shell script",
        }
        label = ext_labels.get(ext, f"{ext.lstrip('.')} file")
        lines = content.count("\n") + 1
        chars = len(content)
        return (
            f"{label} ({lines} lines, {chars:,} chars)",
            self._truncate(content),
        )

    def _truncate(self, text: str) -> str:
        if len(text) <= self.MAX_CHARS:
            return text
        half = self.MAX_CHARS // 2
        return (
            text[:half]
            + f"\n\n[... {len(text) - self.MAX_CHARS:,} characters omitted — document truncated to fit context ...]\n\n"
            + text[-half:]
        )


# =============================================================================
# SYSTEM PROMPT BUILDER
# =============================================================================

def build_system_prompt(doc_label: str | None, doc_content: str | None) -> str:
    """Build the LLM system prompt with optional document context injected."""

    base = textwrap.dedent("""
        You are a voice AI assistant built by Quantum Solutions Ltd, based in Kampala, Uganda.
        You are running on LiveKit and speaking to the user via voice.

        VOICE RULES — critical:
        - Respond in natural spoken English only. No markdown, no bullet symbols, no asterisks,
          no numbered lists with dots. Structure with spoken transitions: "First... then... finally..."
        - Keep each response under 4 sentences unless the user asks for more detail.
        - For complex answers, offer to go deeper: "Want me to explain more about that part?"
        - Spell out abbreviations on first use: "R L S — Row Level Security"
        - When referencing page numbers or sections, say them naturally:
          "On page three..." not "[Page 3]"
        - Never say "As an AI language model" or similar disclaimers.
        - If you don't know something, say so clearly and briefly.

        EXPERTISE:
        - PDF documents: reports, proposals, technical manuals
        - SQL schemas: tables, foreign keys, RLS policies, RPCs, triggers, indexes
        - Markdown specs: architecture docs, feature specs, READMEs
        - Code files: JavaScript, TypeScript, Python, React
        - Data files: JSON, CSV — structure and content questions
        - Claude AI / LLM session transcripts: decisions, actions, conclusions

        When answering questions about a loaded document:
        - Reference specific content from it ("The schema shows a table called...")
        - Be precise about page numbers, table names, policy names, function names
        - For SQL, explain RLS policies in plain language ("This policy means only the
          tenant who owns the row can see it")
        - For PDFs, summarise sections on request or dive into specific pages
    """).strip()

    if doc_label and doc_content:
        doc_section = textwrap.dedent(f"""

        ════════════════════════════════════════
        LOADED DOCUMENT: {doc_label}
        ════════════════════════════════════════
        The following is the full extracted content of the document the user wants
        to discuss. Use it to answer their questions accurately.

        {doc_content}

        ════════════════════════════════════════
        END OF DOCUMENT
        ════════════════════════════════════════

        The user may ask you to summarise, explain sections, find specific information,
        compare parts, or answer detailed questions. Always ground your answers in the
        document content above.
        """)
        return base + doc_section
    else:
        return base + "\n\nNo document is currently loaded. Answer from your general knowledge."


# =============================================================================
# AGENT ENTRYPOINT
# =============================================================================

# Parse --doc argument before LiveKit CLI takes over
_DOC_PATH: str | None = None
_doc_label: str | None = None
_doc_content: str | None = None

def _parse_doc_arg():
    global _DOC_PATH, _doc_label, _doc_content
    args = sys.argv[:]
    if "--doc" in args:
        idx = args.index("--doc")
        if idx + 1 < len(args):
            _DOC_PATH = args[idx + 1]
            # Remove from sys.argv so LiveKit CLI doesn't choke on it
            sys.argv.pop(idx)      # remove --doc
            sys.argv.pop(idx)      # remove the value

    if _DOC_PATH:
        reader = DocumentReader()
        try:
            _doc_label, _doc_content = reader.read(_DOC_PATH)
            logger.info(f"Document loaded: {_doc_label}")
        except Exception as e:
            logger.error(f"Failed to load document: {e}")
            sys.exit(1)


def prewarm(proc: JobProcess):
    """Preload the Silero VAD model in the worker process."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main agent job — connects to LiveKit room and starts voice pipeline."""

    # Build system prompt (with or without doc)
    system_prompt = build_system_prompt(_doc_label, _doc_content)

    if _doc_label:
        initial_greeting = (
            f"Hello! I've loaded your document — {_doc_label}. "
            "What would you like to know about it?"
        )
    else:
        initial_greeting = (
            "Hello! I'm your Quantum Solutions voice assistant. "
            "I can answer questions about documents, SQL schemas, code, or anything you need. "
            "How can I help you today?"
        )

    logger.info(f"Agent starting in room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for a participant to join
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # ── Build LLM ────────────────────────────────────────────────────────────
    agent_llm = anthropic.LLM(
        model="claude-sonnet-4-6",
        # System prompt carries the document context
    )

    # ── Build TTS ────────────────────────────────────────────────────────────
    if TTS_PROVIDER == "elevenlabs":
        tts = tts_plugin.TTS(
            voice_id="ErXwobaYiN019PkySvjV",   # "Antoni" — clear, professional male
            model_id="eleven_turbo_v2",
        )
    elif TTS_PROVIDER == "cartesia":
        tts = tts_plugin.TTS(
            model="sonic-english",
            voice=tts_plugin.Voice(
                id="a0e99841-438c-4a64-b679-ae501e7d6091",  # "Barbershop Man"
                embedding=tts_plugin.VoiceSettings(speed=1.0, emotion=[]),
            ),
        )
    else:
        raise RuntimeError(
            "No TTS provider available. Install one:\n"
            "  pip install livekit-plugins-elevenlabs\n"
            "  pip install livekit-plugins-cartesia"
        )

    # ── Build STT ────────────────────────────────────────────────────────────
    stt = deepgram.STT(
        model="nova-2",
        language="en",
        smart_format=True,
    )

    # ── Voice Pipeline ────────────────────────────────────────────────────────
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=stt,
        llm=agent_llm,
        tts=tts,
        chat_ctx=llm.ChatContext().append(
            role="system",
            text=system_prompt,
        ),
        # Tuning
        min_endpointing_delay=0.5,   # seconds of silence before treating as end of utterance
        max_endpointing_delay=6.0,   # max wait even with background noise
        allow_interruptions=True,
        interrupt_speech_duration=0.5,
        interrupt_min_words=0,
        preemptive_synthesis=True,   # start generating TTS before STT fully finalises
    )

    # ── Metrics logging ──────────────────────────────────────────────────────
    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def on_metrics(mtrx: metrics.AgentMetrics):
        metrics.log_metrics(mtrx)
        usage_collector.collect(mtrx)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Session usage summary: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # ── Start ─────────────────────────────────────────────────────────────────
    agent.start(ctx.room, participant)
    await agent.say(initial_greeting, allow_interruptions=True)

    # Keep the job alive
    await asyncio.sleep(float("inf"))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    _parse_doc_arg()

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )


# =============================================================================
# .env.example  (paste into a file named .env and fill in your keys)
# =============================================================================
"""
# .env.example
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=APIxxxxxxxxxxxxxxxxxx
LIVEKIT_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxx
DEEPGRAM_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ELEVENLABS_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# CARTESIA_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # alternative TTS
"""


# =============================================================================
# REQUIREMENTS  (paste into requirements.txt)
# =============================================================================
"""
# requirements.txt
livekit-agents[deepgram,anthropic,elevenlabs,silero]~=1.0
pdfplumber>=0.10
pypdf>=4.0
pytesseract>=0.3
Pillow>=10.0
python-dotenv>=1.0
"""
