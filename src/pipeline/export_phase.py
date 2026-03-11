"""Phase 1: Export from SQLite to DuckDB"""

import sqlite3
from pathlib import Path
from typing import Dict
from rich.progress import Progress, TaskID


class ExportPhase:
    """Export messages and sessions from SQLite"""
    
    def __init__(self, sqlite_path: str, duckdb_server):
        self.sqlite_path = Path(sqlite_path)
        self.duckdb = duckdb_server
        
    def run(self, progress: Progress, task: TaskID) -> Dict[str, int]:
        """Export all data to DuckDB"""
        
        if not self.sqlite_path.exists():
            raise FileNotFoundError(f"SQLite not found: {self.sqlite_path}")
            
        # Connect to SQLite
        sqlite_conn = sqlite3.connect(str(self.sqlite_path))
        sqlite_conn.row_factory = sqlite3.Row
        
        try:
            # Export sessions
            cur = sqlite_conn.execute("SELECT id, title, created_at FROM chat_sessions")
            sessions = [tuple(row) for row in cur.fetchall()]
            
            self.duckdb.execute("BEGIN TRANSACTION")
            for session in sessions:
                self.duckdb.execute(
                    "INSERT INTO sessions_export VALUES (?, ?, ?)",
                    session
                )
            self.duckdb.execute("COMMIT")
            
            # Export messages
            cur = sqlite_conn.execute("""
                SELECT m.id, m.session_id, m.role, m.content, m.created_at
                FROM messages m
                ORDER BY m.session_id, m.created_at
            """)
            
            batch = []
            total = 0
            
            while True:
                rows = cur.fetchmany(1000)
                if not rows:
                    break
                    
                batch.extend([tuple(row) for row in rows])
                total += len(rows)
                
                if len(batch) >= 5000:
                    self._insert_batch(batch)
                    batch = []
                    progress.update(task, completed=total)
                    
            # Insert remaining
            if batch:
                self._insert_batch(batch)
                progress.update(task, completed=total)
                
        finally:
            sqlite_conn.close()
            
        # Return stats
        stats = self.duckdb.get_stats()
        return {
            'sessions': stats.get('sessions_count', 0),
            'messages': stats.get('messages_count', 0)
        }
        
    def _insert_batch(self, batch):
        """Insert batch to DuckDB"""
        self.duckdb.execute("BEGIN TRANSACTION")
        for row in batch:
            self.duckdb.execute(
                "INSERT INTO messages_export VALUES (?, ?, ?, ?, ?)",
                row
            )
        self.duckdb.execute("COMMIT")
