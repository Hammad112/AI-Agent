"""
main.py
-------
CLI entry point for the Generic PDF AI Customer Service Agent.

Usage:
    python main.py                    # will prompt you to enter the PDF path
    python main.py --pdf salon.pdf    # or pass it directly

Workflow:
    1. Load environment variables
    2. Initialise SQLite database
    3. Process PDF → intelligent chunks
    4. Detect business type via LLM
    5. Enrich knowledge base via LLM
    6. Generate synthetic customer/order data
    7. CLI login or register
    8. Run the LangGraph agent conversation loop
"""

import os
import sys
import uuid
import argparse
from pathlib import Path
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich import print as rprint

# ── Load .env ───────────────────────────────────
load_dotenv()

# ── Imports after env load ───────────────────────
from core.database import init_db, generate_synthetic_data, set_business_meta, get_business_meta
from processing.pdf_processor import process_pdf, load_chunks_from_db, chunks_already_stored, get_pdf_summary
from processing.knowledge_enricher import (
    detect_business_type,
    enrich_knowledge,
    enrichment_already_done,
    load_enriched_from_db,
)
from auth.auth import authenticate
from agent.agent import run_agent_turn

console = Console()


# ─────────────────────────────────────────────
# CLI Banner
# ─────────────────────────────────────────────

def print_banner(business_name: str, business_type: str) -> None:
    console.print(
        Panel.fit(
            f"[bold cyan]🤖 AI Customer Service Assistant[/]\n"
            f"[white]Business:[/] [bold yellow]{business_name}[/]\n"
            f"[white]Type:[/] [dim]{business_type}[/]\n\n"
            f"[dim]Type your message, or 'exit' / 'quit' to leave.[/]",
            border_style="cyan",
            title="[bold]GenericAgent[/]",
            subtitle="[dim]Powered by Gemini + LangGraph[/]",
        )
    )


def print_step(icon: str, msg: str) -> None:
    console.print(f"  {icon} [bold]{msg}[/]")


# ─────────────────────────────────────────────
# Startup Pipeline
# ─────────────────────────────────────────────

def startup(pdf_path: str) -> tuple[str, str, list[dict]]:
    """
    Run the full startup pipeline:
      DB init → PDF processing → business detection →
      knowledge enrichment → synthetic data generation.
    Returns (business_name, business_type, all_chunks).
    """
    console.rule("[bold cyan]Starting up Agent[/]")

    # 1. Init DB
    print_step("🗄️ ", "Initialising database…")
    init_db()

    # 2. Process PDF (or load from DB if already done)
    if chunks_already_stored() and get_business_meta("business_name"):
        print_step("📄", "PDF already processed — loading from database…")
        pdf_chunks = load_chunks_from_db()
        business_name = get_business_meta("business_name") or "The Business"
        business_type = get_business_meta("business_type") or "general business"
    else:
        print_step("📄", f"Processing PDF: [cyan]{pdf_path}[/]…")
        pdf_chunks = process_pdf(pdf_path)
        console.print(f"     [green]✓[/] {len(pdf_chunks)} chunks created")

        # 3. Detect business type
        print_step("🔍", "Detecting business type…")
        pdf_summary = get_pdf_summary(pdf_chunks)
        business_name, business_type = detect_business_type(pdf_summary)
        console.print(f"     [green]✓[/] Detected: [bold yellow]{business_name}[/] ({business_type})")

        # Store in DB
        set_business_meta("business_name", business_name)
        set_business_meta("business_type", business_type)
        set_business_meta("pdf_path", str(pdf_path))

    # 4. Enrich knowledge
    if enrichment_already_done():
        print_step("🧠", "Knowledge enrichment already done — loading…")
        enriched = load_enriched_from_db()
    else:
        print_step("🧠", "Enriching knowledge base with LLM…")
        enriched = enrich_knowledge(pdf_chunks, business_name, business_type)
        console.print(f"     [green]✓[/] {len(enriched)} knowledge articles created")

    # 5. Generate synthetic data
    print_step("🎲", "Generating synthetic business data…")
    generate_synthetic_data(business_type, business_name)
    console.print(f"     [green]✓[/] Customers, providers, services, orders ready")

    # Combine all chunks for the RAG engine
    # Normalise enriched chunks to have the same keys as pdf_chunks
    normalised_enriched = [
        {
            "chunk_index": 10000 + i,
            "page_num": 0,
            "section_title": e.get("topic", "Enriched"),
            "text": e.get("content", ""),
            "topic_tags": e.get("topic_tags"),
            "topic": e.get("topic", ""),
            "id": e.get("id", 10000 + i),
        }
        for i, e in enumerate(enriched)
    ]
    all_chunks = pdf_chunks + normalised_enriched

    console.print(
        f"\n  [bold green]✓ Ready![/] {len(pdf_chunks)} PDF chunks + "
        f"{len(enriched)} enriched articles = [bold]{len(all_chunks)} total knowledge items[/]\n"
    )
    return business_name, business_type, all_chunks


# ─────────────────────────────────────────────
# Conversation Loop
# ─────────────────────────────────────────────

def conversation_loop(
    user: dict,
    business_name: str,
    business_type: str,
    all_chunks: list[dict],
) -> None:
    """Run the interactive conversation loop with the agent."""
    conversation_id = str(uuid.uuid4())
    print_banner(business_name, business_type)
    console.print(
        f"[dim]Session ID:[/] [cyan]{conversation_id[:8]}…[/]  "
        f"[dim]Logged in as:[/] [bold]{user.get('full_name') or user.get('username')}[/]\n"
    )

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! 👋[/]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye", "goodbye"}:
            console.print("\n[bold cyan]Thank you for chatting! Have a great day! 👋[/]")
            break

        with console.status("[dim]Thinking…[/]", spinner="dots"):
            try:
                response = run_agent_turn(
                    user_id=user["id"],
                    conversation_id=conversation_id,
                    query=user_input,
                    all_chunks=all_chunks,
                    business_name=business_name,
                    business_type=business_type,
                )
            except Exception as e:
                console.print(f"[red]Agent error:[/] {e}")
                continue

        console.print(
            Panel(
                f"[white]{response}[/]",
                title=f"[bold cyan]🤖 {business_name} Assistant[/]",
                border_style="dim cyan",
                padding=(1, 2),
            )
        )
        console.print()


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def _prompt_for_pdf() -> Path:
    """Interactively ask the user to provide a PDF file path."""
    console.print(
        Panel.fit(
            "[bold cyan]🤖 Generic PDF AI Customer Service Agent[/]\n\n"
            "[white]This agent reads any business PDF and becomes\n"
            "a smart customer service assistant for that business.[/]\n\n"
            "[dim]Examples: salon menu, hotel brochure, clinic info,\n"
            "car rental guide, restaurant menu, etc.[/]",
            border_style="cyan",
            title="[bold]Welcome[/]",
        )
    )
    console.print()

    while True:
        pdf_input = Prompt.ask(
            "[bold yellow]📄 Enter the path to your business PDF file[/]\n"
            "  [dim](You can drag and drop the file into this terminal)[/]\n"
            "  [bold green]PDF path[/]"
        ).strip()

        # Strip PowerShell drag-drop prefix: & 'path' or & "path"
        if pdf_input.startswith("& "):
            pdf_input = pdf_input[2:].strip()
        # Strip surrounding quotes (single or double)
        pdf_input = pdf_input.strip('"').strip("'").strip()

        if not pdf_input:
            console.print("[red]Please enter a file path.[/]\n")
            continue

        pdf_path = Path(pdf_input)
        if not pdf_path.exists():
            console.print(
                f"[red]✗ File not found:[/] [yellow]{pdf_path}[/]\n"
                f"  Please check the path and try again.\n"
            )
            continue
        if pdf_path.suffix.lower() != ".pdf":
            console.print(
                f"[red]✗ That doesn't look like a PDF file:[/] [yellow]{pdf_path.name}[/]\n"
                f"  Please provide a file ending in [bold].pdf[/]\n"
            )
            continue

        console.print(f"\n  [green]✓[/] PDF found: [bold]{pdf_path.name}[/] ({pdf_path.stat().st_size // 1024} KB)\n")
        return pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Generic PDF AI Customer Service Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # interactive — will ask for PDF
  python main.py --pdf salon.pdf
  python main.py --pdf hotel_info.pdf
        """,
    )
    parser.add_argument(
        "--pdf",
        required=False,       # now optional
        type=str,
        default=None,
        help="Path to the business PDF file (optional — will prompt if not given)",
    )
    args = parser.parse_args()

    # ── Determine PDF path ──────────────────────────────────
    if args.pdf:
        # Passed via command line
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            console.print(f"[red]Error: PDF file not found:[/] {pdf_path}")
            sys.exit(1)
    else:
        # Interactive prompt
        pdf_path = _prompt_for_pdf()

    # ── Check API keys ──────────────────────────────────────
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[red bold]Error:[/] No API keys found.\n"
            "Open [cyan].env[/] and add your GEMINI_API_KEY or OPENAI_API_KEY."
        )
        sys.exit(1)

    try:
        # Run startup pipeline
        business_name, business_type, all_chunks = startup(str(pdf_path))

        # Authentication
        console.rule("[bold cyan]Authentication[/]")
        user = authenticate()

        # Start conversation
        conversation_loop(user, business_name, business_type, all_chunks)

    except KeyboardInterrupt:
        console.print("\n[dim]Session ended.[/]")
    except Exception as e:
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
