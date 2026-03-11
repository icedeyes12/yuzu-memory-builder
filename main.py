#!/usr/bin/env python3
"""
Yuzu Memory Builder - Interactive CLI
Single entry point for all memory rebuild operations
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.config import Config
from core.pipeline import MemoryPipeline
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


async def main():
    """Main entry point"""
    console.print(Panel.fit(
        "[bold magenta]🌸 Yuzu Memory Builder[/bold magenta]\n"
        "[dim]Rebuild ~7k messages with semantic memory layer[/dim]",
        border_style="magenta"
    ))
    
    # Load config
    config = Config()
    
    # Show config
    console.print(f"[dim]Model:[/dim] {config.model_name}")
    console.print(f"[dim]Batch:[/dim] {config.batch_size}")
    console.print()
    
    # Select phases
    phases = []
    
    console.print("[bold]Select phases to run:[/bold]")
    console.print("  [1] Export - Pull history from Supabase")
    console.print("  [2] Embed - Generate embeddings (takes ~1 hour)")
    console.print("  [3] Preview - Validate before migration")
    console.print("  [4] Migrate - Inject to Supabase")
    console.print()
    
    # Default: run all
    response = console.input("[bold cyan]Run all phases? (Y/n):[/bold cyan] ").strip().lower()
    
    if response in ('', 'y', 'yes'):
        phases = ['export', 'embed', 'validate', 'migrate']
    else:
        if console.input("  Include export? (y/n): ").strip().lower() == 'y':
            phases.append('export')
        if console.input("  Include embed? (y/n): ").strip().lower() == 'y':
            phases.append('embed')
        if console.input("  Include validate? (y/n): ").strip().lower() == 'y':
            phases.append('validate')
        if console.input("  Include migrate? (y/n): ").strip().lower() == 'y':
            phases.append('migrate')
    
    if not phases:
        console.print("[yellow]No phases selected. Exiting.[/yellow]")
        return
    
    # Run pipeline
    console.print(f"\n[bold green]Running phases: {', '.join(phases)}[/bold green]\n")
    
    try:
        async with MemoryPipeline(config) as pipeline:
            await pipeline.run(phases)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    asyncio.run(main())
