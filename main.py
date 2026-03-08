"""
main.py
-------
Entry point for the Generic AI Customer Service Agent.
Handles: PDF input, startup pipeline, authentication, conversation loop.

Usage:
    python main.py                        # interactive — will ask for PDF
    python main.py --pdf salon.pdf
    python main.py --pdf sample_pdfs/mario_pizza.md
"""

import argparse
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from dotenv import load_dotenv

load_dotenv()

from core.database import (
    init_db,
    generate_synthetic_data,
    set_business_meta,
    get_business_meta,
)
from processing.pdf_processor import process_pdf, load_chunks, save_chunks_to_db
from processing.knowledge_enricher import detect_business_type, enrich_knowledge
from auth.auth import authenticate
from agent.agent import run_agent_turn

console = Console()


# ── Startup Pipeline ──────────────────────────

def startup(file_path: str) -> tuple[str, str, list[dict]]:
    """
    Full startup pipeline:
      1. Init database
      2. Process business document (PDF/MD/TXT)
      3. Detect business type
      4. Enrich knowledge via LLM
      5. Generate synthetic data
    """
    # 1. Database
    with console.status("[bold green]Initializing database..."):
        init_db()

    # Check if already processed
    existing_name = get_business_meta("business_name")
    if existing_name:
        console.print(f"[yellow]Business already loaded:[/] {existing_name} (Database: {os.getenv('DB_NAME')})")
        answer = Prompt.ask("Reload from file?", choices=["y", "n"], default="n")
        if answer == "n":
            with console.status("[bold green]Loading existing business data..."):
                chunks = load_chunks(existing_name)
                btype = get_business_meta("business_type") or "general"
                return existing_name, btype, chunks

    with console.status("[bold green]Starting up..."):
        # 2. Process document
        console.print(f"[cyan]Processing:[/] {file_path}")
        all_chunks = process_pdf(file_path)
        console.print(f"  → Created [green]{len(all_chunks)}[/] knowledge chunks")

        # 3. Detect business type
        sample_text = "\n".join(c.get("text", "")[:200] for c in all_chunks[:5])
        with console.status("[bold]Detecting business type..."):
            business_info = detect_business_type(sample_text)
        business_name = business_info.get("business_name", "Business")
        business_type = business_info.get("business_type", "general")
        console.print(f"  → Detected: [bold]{business_name}[/] ({business_type})")

        set_business_meta("business_name", business_name)
        set_business_meta("business_type", business_type)
        
        # Save chunks now that we know the business name
        save_chunks_to_db(all_chunks, business_name)

        # 4. Enrich knowledge
        with console.status("[bold]Enriching knowledge base..."):
            enrich_knowledge(sample_text, business_type, business_name)
        console.print("  → Knowledge enrichment [green]complete[/]")

        # 5. Synthetic data
        with console.status("[bold]Generating synthetic data..."):
            generate_synthetic_data(business_type, business_name, all_chunks)
        console.print("  → Synthetic data [green]generated[/]")

        # Store business hours if provided
        hours_meta = get_business_meta("business_hours")
        if not hours_meta:
            # Try to find hours in chunks
            for chunk in all_chunks:
                if "hour" in chunk.get("section_title", "").lower() or "hour" in chunk.get("text", "").lower()[:50]:
                    set_business_meta("business_hours", chunk.get("text", "")[:500])
                    break

        return business_name, business_type, all_chunks


# ── Interactive PDF prompt ────────────────────

def _prompt_for_pdf() -> Path:
    """Interactively ask the user for a PDF or business document path."""
    console.print(
        Panel(
            "[bold cyan]Welcome to the Generic AI Customer Service Agent[/]\n\n"
            "This agent can handle orders, appointments, and customer service\n"
            "for [bold]any business[/] — just provide a description file.\n\n"
            "Supported formats: [green].pdf[/], [green].md[/], [green].txt[/]",
            title="🤖 AI Agent",
            border_style="cyan",
        )
    )

    while True:
        choice = Prompt.ask(
            "Enter the path to your business description file (e.g., salon.pdf or info.md)"
        )

        path = Path(choice.strip())
        if path.exists() and path.is_file():
            return path
        
        if not path.exists():
            console.print(f"[red]Error:[/] File not found at [bold]{path}[/]. Please check the path and try again.")
        else:
            console.print(f"[red]Error:[/] [bold]{path}[/] is not a valid file. Please provide a path to a .pdf, .md, or .txt file.")


# ── Conversation loop ─────────────────────────

def conversation_loop(
    user: dict,
    business_name: str,
    business_type: str,
    all_chunks: list[dict],
) -> None:
    """Main interactive conversation loop."""
    conversation_id = str(uuid.uuid4())
    user_id = user["id"]

    console.print()
    console.rule(f"[bold green]Chat with {business_name} AI Agent[/]")
    console.print(
        "[dim]Type your message below. Type 'quit' or 'exit' to end.\n"
        "Type 'history' to see your order history.\n"
        "Type 'cart' to see your cart.\n"
        "Type 'clear' to start a new conversation.[/]\n"
    )

    while True:
        try:
            user_input = Prompt.ask("[bold blue]You[/]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/]")
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip().lower()
        if cmd in ("quit", "exit", "bye"):
            console.print("[dim]Goodbye! 👋[/]")
            break
        elif cmd == "clear":
            conversation_id = str(uuid.uuid4())
            console.print("[yellow]New conversation started.[/]")
            continue

        # Run agent
        with console.status("[dim]Thinking...[/]"):
            try:
                agent_out = run_agent_turn(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    query=user_input,
                    all_chunks=all_chunks,
                    business_name=business_name,
                    business_type=business_type,
                )
                response_text = agent_out.get("response", "")
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")
                continue

        console.print(f"\n[bold green]Agent:[/] {response_text}\n")
        
        if "ics_file_url" in agent_out and agent_out["ics_file_url"]:
            console.print(f"[bold yellow]Calendar Event Generated:[/] {agent_out['ics_file_url']}\n")


# ── Main ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generic AI Customer Service Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                # interactive — will ask for file
  python main.py --pdf sample_pdfs/mario_pizza.md
  python main.py --pdf sample_pdfs/bright_smile_dental.md
  python main.py --pdf path/to/your_business.pdf
        """,
    )
    parser.add_argument(
        "--pdf",
        required=False,
        type=str,
        default=None,
        help="Path to the business description file (.pdf, .md, .txt)",
    )
    args = parser.parse_args()

    # ── Determine file path ─────────────────────
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            console.print(f"[red]Error: File not found:[/] {pdf_path}")
            sys.exit(1)
    else:
        pdf_path = _prompt_for_pdf()

    # Check API keys
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
