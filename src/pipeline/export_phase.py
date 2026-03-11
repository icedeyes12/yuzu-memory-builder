"""Phase 1: Export from Supabase to DuckDB"""

from typing import Dict
from rich.progress import Progress, TaskID
from ..core.supabase_client import SupabaseClient
from ..core.duckdb_server import DuckDBServer


class ExportPhase:
    """Export messages and sessions from Supabase to local DuckDB"""
    
    def __init__(self, supabase_client: SupabaseClient, duckdb_server: DuckDBServer):
        self.supabase = supabase_client
        self.duckdb = duckdb_server
        
    def run(self, progress: Progress, task: TaskID) -> Dict[str, int]:
        """Export all data to DuckDB"""
        
        # Fetch sessions
        print("📦 Fetching sessions from Supabase...")
        sessions = self.supabase.fetch_sessions()
        
        # Insert to DuckDB
        self.duckdb.execute("BEGIN TRANSACTION")
        for session in sessions:
            self.duckdb.execute(
                "INSERT INTO sessions_export VALUES (?, ?, ?, ?)",
                (session['id'], session['user_id'], session['title'], session['created_at'])
            )
        self.duckdb.execute("COMMIT")
        
        # Fetch messages in batches
        print("📦 Fetching messages from Supabase...")
        total_messages = 0
        batch_size = 500
        
        # Get total count first
        count_result = self.supabase.fetch_messages(limit=1, offset=0)
        estimated_total = self.supabase.get_messages_count()
        
        progress.update(task, total=estimated_total)
        
        offset = 0
        while True:
            messages = self.supabase.fetch_messages(limit=batch_size, offset=offset)
            if not messages:
                break
            
            # Insert batch
            self.duckdb.execute("BEGIN TRANSACTION")
            for msg in messages:
                self.duckdb.execute(
                    "INSERT INTO messages_export VALUES (?, ?, ?, ?, ?, ?)",
                    (msg['id'], msg['session_id'], msg['user_id'], 
                     msg['role'], msg['content'], msg['created_at'])
                )
            self.duckdb.execute("COMMIT")
            
            total_messages += len(messages)
            offset += batch_size
            progress.update(task, completed=min(total_messages, estimated_total))
            
            if len(messages) < batch_size:
                break
                
        # Return stats
        stats = self.duckdb.get_stats()
        return {
            'sessions': stats.get('sessions_count', 0),
            'messages': stats.get('messages_count', 0)
        }
