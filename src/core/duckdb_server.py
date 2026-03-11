"""DuckDB local storage"""
import duckdb
from pathlib import Path
from typing import Dict, List, Any, Optional


class DuckDBServer:
    """Local DuckDB for intermediate storage"""
    
    def __init__(self, db_path: str = "./yuzu_local.duckdb"):
        self.db_path = Path(db_path)
        self.conn = None
        
    def connect(self):
        """Connect to DuckDB"""
        self.conn = duckdb.connect(str(self.db_path))
        self._create_tables()
        print(f"✅ Connected to DuckDB: {self.db_path}")
        
    def _create_tables(self):
        """Create storage tables"""
        # Messages export
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages_export (
                id INTEGER,
                session_id INTEGER,
                role VARCHAR,
                content TEXT,
                created_at TIMESTAMP
            )
        """)
        
        # Sessions export
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions_export (
                id INTEGER,
                title VARCHAR,
                created_at TIMESTAMP
            )
        """)
        
        # Memories with embeddings
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories_with_embeddings (
                id INTEGER,
                user_id INTEGER,
                session_id INTEGER,
                content TEXT,
                embedding FLOAT[],
                memory_type VARCHAR,
                importance INTEGER,
                created_at TIMESTAMP
            )
        """)
        
        self.conn.commit()
        
    def execute(self, query: str, params: list = None):
        """Execute query"""
        if params:
            return self.conn.execute(query, params)
        return self.conn.execute(query)
        
    def fetch_all(self, query: str) -> List[Any]:
        """Fetch all results"""
        return self.conn.execute(query).fetchall()
        
    def get_message_count(self) -> int:
        """Get total messages"""
        result = self.conn.execute("SELECT COUNT(*) FROM messages_export").fetchone()
        return result[0] if result else 0
        
    def get_session_count(self) -> int:
        """Get total sessions"""
        result = self.conn.execute("SELECT COUNT(*) FROM sessions_export").fetchone()
        return result[0] if result else 0
        
    def get_stats(self) -> Dict[str, int]:
        """Get storage stats"""
        return {
            'messages_count': self.get_message_count(),
            'sessions_count': self.get_session_count(),
            'memories_count': self.fetch_all("SELECT COUNT(*) FROM memories_with_embeddings")[0][0]
        }
        
    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()
            print("🔌 DuckDB closed")
