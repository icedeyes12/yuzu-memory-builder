"""Memory rebuild pipeline"""
import asyncio
from typing import List, Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.config import Config
from core.duckdb_server import DuckDBServer
from core.supabase_client import SupabaseClient
from core.onnx_server import get_onnx_server

console = Console()


class MemoryPipeline:
    """Pipeline for rebuilding memory layer"""
    
    def __init__(self, config: Config):
        self.config = config
        self.duckdb = None
        self.supabase = None
        self.onnx = None
        
    async def __aenter__(self):
        """Setup connections"""
        # DuckDB
        self.duckdb = DuckDBServer(config.local_db_path)
        self.duckdb.connect()
        
        # Supabase
        if config.database_url:
            self.supabase = SupabaseClient(config.database_url)
            self.supabase.connect()
            
        # ONNX
        self.onnx = get_onnx_server(config.model_name)
        self.onnx.start()
        
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup"""
        if self.duckdb:
            self.duckdb.close()
        if self.supabase:
            self.supabase.disconnect()
        if self.onnx:
            self.onnx.stop()
            
    async def run(self, phases: List[str]):
        """Run selected phases"""
        for phase in phases:
            console.print(f"\n[bold cyan]=== Phase: {phase.upper()} ===[/bold cyan]")
            
            if phase == 'export':
                await self.export()
            elif phase == 'embed':
                await self.embed()
            elif phase == 'validate':
                await self.validate()
            elif phase == 'migrate':
                await self.migrate()
                
        console.print("\n[bold green]✅ All phases complete![/bold green]")
        
    async def export(self):
        """Export from Supabase to DuckDB"""
        if not self.supabase:
            console.print("[yellow]No Supabase connection. Skipping.[/yellow]")
            return
            
        console.print("📤 Exporting messages from Supabase...")
        
        # Get user
        user = self.supabase.get_user(user_id=1)
        if not user:
            console.print("[red]No user found![/red]")
            return
            
        console.print(f"👤 User: {user.get('email')} ({user.get('display_name')})")
        
        # Export sessions
        sessions = self.supabase.fetch_sessions()
        console.print(f"📚 Sessions: {len(sessions)}")
        
        for s in sessions:
            self.duckdb.execute(
                "INSERT INTO sessions_export VALUES (?, ?, ?)",
                [s['id'], s['title'], s['created_at']]
            )
            
        # Export messages in batches
        offset = 0
        batch_size = 1000
        total = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Exporting messages...", total=None)
            
            while True:
                messages = self.supabase.fetch_messages(limit=batch_size, offset=offset)
                if not messages:
                    break
                    
                for m in messages:
                    self.duckdb.execute(
                        "INSERT INTO messages_export VALUES (?, ?, ?, ?, ?)",
                        [m['id'], m['session_id'], m['role'], m['content'], m['created_at']]
                    )
                    
                total += len(messages)
                offset += batch_size
                progress.update(task, description=f"Exported {total} messages...")
                
        stats = self.duckdb.get_stats()
        console.print(f"✅ Exported: {stats['messages_count']} messages, {stats['sessions_count']} sessions")
        
    async def embed(self):
        """Generate embeddings"""
        console.print("🧠 Generating embeddings...")
        
        messages = self.duckdb.fetch_all("SELECT * FROM messages_export")
        total = len(messages)
        
        console.print(f"📝 Processing {total} messages...")
        
        with Progress(console=console) as progress:
            task = progress.add_task("Embedding...", total=total)
            
            batch = []
            for i, msg in enumerate(messages):
                content = msg[3]  # content column
                if content and len(content) > 10:  # Skip short messages
                    batch.append({
                        'id': msg[0],
                        'session_id': msg[1],
                        'role': msg[2],
                        'content': content,
                        'created_at': msg[4]
                    })
                    
                if len(batch) >= self.config.batch_size:
                    # Generate embeddings
                    texts = [b['content'] for b in batch]
                    embeddings = self.onnx.embed(texts)
                    
                    # Store with embeddings
                    for b, emb in zip(batch, embeddings):
                        self.duckdb.execute(
                            "INSERT INTO memories_with_embeddings VALUES (?, ?, ?, ?, ?, ?, ?)",
                            [b['id'], 1, b['session_id'], b['content'], emb, 'episodic', 50]
                        )
                        
                    batch = []
                    
                progress.update(task, advance=1)
                
        stats = self.duckdb.get_stats()
        console.print(f"✅ Embedded: {stats.get('memories_count', 0)} memories")
        
    async def validate(self):
        """Validate before migration"""
        console.print("🔍 Validating...")
        
        stats = self.duckdb.get_stats()
        console.print(f"📊 Local stats: {stats}")
        
        if self.supabase:
            supabase_stats = self.supabase.get_stats()
            console.print(f"📊 Supabase stats: {supabase_stats}")
            
        console.print("✅ Validation complete")
        
    async def migrate(self):
        """Migrate to Supabase"""
        if not self.supabase:
            console.print("[red]No Supabase connection![/red]")
            return
            
        console.print("🚀 Migrating to Supabase...")
        
        memories = self.duckdb.fetch_all("SELECT * FROM memories_with_embeddings")
        
        if not memories:
            console.print("[yellow]No memories to migrate![/yellow]")
            return
            
        console.print(f"📦 Migrating {len(memories)} memories...")
        
        # Convert to dicts
        mem_dicts = []
        for m in memories:
            mem_dicts.append({
                'user_id': m[1],
                'session_id': m[2],
                'content': m[3],
                'embedding': m[4],
                'memory_type': m[5],
                'importance': m[6]
            })
            
        inserted = self.supabase.migrate_memories(mem_dicts)
        
        console.print(f"✅ Migrated: {inserted} memories")


# Entry point for import
config = Config()
