#!/usr/bin/env python3
"""
Yuzu Memory Builder - Interactive CLI
Single entry point for all memory rebuilding operations
"""

import os
import sys
import asyncio
import signal
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich import box

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import Config
from src.core.duckdb_server import DuckDBServer
from src.core.onnx_server import ONNXServer
from src.core.supabase_client import SupabaseClient
from src.pipeline.export_phase import ExportPhase
from src.pipeline.embed_phase import EmbedPhase
from src.pipeline.validate_phase import ValidatePhase
from src.pipeline.migrate_phase import MigratePhase

console = Console()


class MemoryBuilderApp:
    """Main application orchestrator"""
    
    def __init__(self):
        self.config = Config.from_env()
        self.duckdb: Optional[DuckDBServer] = None
        self.onnx: Optional[ONNXServer] = None
        self.supabase: Optional[SupabaseClient] = None
        self.running = False
        
    async def __aenter__(self):
        """Async context manager - start services"""
        console.print("[dim]Starting services...[/dim]")
        
        # Start DuckDB
        self.duckdb = DuckDBServer(self.config.local_db_path)
        await self.duckdb.start()
        console.print("[green]✓[/green] DuckDB ready")
        
        # Start ONNX (lazy load on first use)
        self.onnx = ONNXServer(self.config.model_name)
        console.print("[green]✓[/green] ONNX server ready")
        
        # Connect Supabase (read-only mode default)
        self.supabase = SupabaseClient(self.config.database_url, read_only=True)
        console.print("[green]✓[/green] Supabase connected (read-only)")
        
        self.running = True
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup services"""
        console.print("\n[dim]Shutting down...[/dim]")
        if self.duckdb:
            await self.duckdb.stop()
        if self.onnx:
            await self.onnx.stop()
        console.print("[green]✓[/green] Clean exit")
        
    def show_banner(self):
        """Display welcome banner"""
        banner = """
[bold magenta]╭──────────────────────────────────────╮[/bold magenta]
[bold magenta]│  🌸 Yuzu Memory Builder v1.0         │[/bold magenta]
[bold magenta]│                                      │[/bold magenta]
[bold magenta]│  Rebuild 7k+ memories with love      │[/bold magenta]
[bold magenta]╰──────────────────────────────────────╯[/bold magenta]
        """
        console.print(banner)
        console.print(f"[dim]Database: {self.config.database_url[:30]}...[/dim]")
        console.print(f"[dim]Local storage: {self.config.local_db_path}[/dim]")
        console.print(f"[dim]Model: {self.config.model_name}[/dim]\n")
        
    def show_menu(self) -> str:
        """Display interactive menu"""
        menu = Table(
            title="[bold cyan]Available Commands[/bold cyan]",
            box=box.ROUNDED,
            show_header=False,
            border_style="cyan"
        )
        menu.add_column(style="green", justify="center")
        menu.add_column(style="white")
        menu.add_column(style="dim")
        
        menu.add_row("1", "export", "Export Supabase → DuckDB")
        menu.add_row("2", "embed", "Generate ONNX embeddings")
        menu.add_row("3", "validate", "Spot-check quality")
        menu.add_row("4", "preview", "Preview memories")
        menu.add_row("5", "migrate", "Batch insert to Supabase")
        menu.add_row("6", "status", "Show current progress")
        menu.add_row("q", "quit", "Exit application")
        
        console.print(menu)
        return Prompt.ask(
            "\n[bold yellow]Select command[/bold yellow]",
            choices=["1", "2", "3", "4", "5", "6", "export", "embed", "validate", "preview", "migrate", "status", "q", "quit"],
            default="status"
        )
        
    async def cmd_export(self):
        """Phase 1: Export data from Supabase"""
        phase = ExportPhase(self.supabase, self.duckdb, console)
        await phase.run()
        
    async def cmd_embed(self):
        """Phase 2: Generate embeddings"""
        phase = EmbedPhase(self.duckdb, self.onnx, console)
        await phase.run()
        
    async def cmd_validate(self):
        """Phase 3: Validate embeddings"""
        phase = ValidatePhase(self.duckdb, self.onnx, console)
        await phase.run()
        
    async def cmd_preview(self):
        """Preview generated memories"""
        table = Table(title="Preview: Top 10 Semantic Memories")
        table.add_column("Fact", style="green", no_wrap=False)
        table.add_column("Category", style="cyan")
        table.add_column("Embedding", style="dim")
        
        rows = self.duckdb.query("""
            SELECT fact, category, length(embedding::TEXT) as emb_len
            FROM semantic_memories_local
            LIMIT 10
        """)
        
        for row in rows:
            table.add_row(row['fact'][:50] + "...", row['category'], f"{row['emb_len']} chars")
            
        console.print(table)
        
    async def cmd_migrate(self):
        """Phase 4: Migrate to Supabase"""
        # Safety check
        if not Confirm.ask("[red]WARNING[/red]: This will WRITE to Supabase. Continue?", default=False):
            console.print("[yellow]Aborted.[/yellow]")
            return
            
        phase = MigratePhase(self.duckdb, self.supabase, console)
        await phase.run()
        
    async def cmd_status(self):
        """Show current status"""
        stats = self.duckdb.get_stats()
        
        status_table = Table(title="[bold]Current Status[/bold]", box=box.ROUNDED)
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Count", justify="right", style="green")
        
        status_table.add_row("Messages exported", str(stats.get('messages_count', 0)))
        status_table.add_row("Sessions exported", str(stats.get('sessions_count', 0)))
        status_table.add_row("Episodic memories", str(stats.get('episodic_count', 0)))
        status_table.add_row("Semantic memories", str(stats.get('semantic_count', 0)))
        status_table.add_row("Embeddings generated", str(stats.get('embedded_count', 0)))
        
        console.print(status_table)
        
    async def run(self):
        """Main loop"""
        self.show_banner()
        
        while self.running:
            try:
                cmd = self.show_menu()
                
                if cmd in ("q", "quit"):
                    break
                    
                # Map numbers to commands
                cmd_map = {
                    "1": "export", "2": "embed", "3": "validate",
                    "4": "preview", "5": "migrate", "6": "status"
                }
                cmd = cmd_map.get(cmd, cmd)
                
                # Execute command
                if cmd == "export":
                    await self.cmd_export()
                elif cmd == "embed":
                    await self.cmd_embed()
                elif cmd == "validate":
                    await self.cmd_validate()
                elif cmd == "preview":
                    await self.cmd_preview()
                elif cmd == "migrate":
                    await self.cmd_migrate()
                elif cmd == "status":
                    await self.cmd_status()
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Use 'quit' to exit.[/yellow]")
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                

async def main():
    """Entry point"""
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        console.print("\n[yellow]Shutting down...[/yellow]")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    async with MemoryBuilderApp() as app:
        await app.run()


if __name__ == "__main__":
    asyncio.run(main())
