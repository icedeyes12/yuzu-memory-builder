"""DuckDB local storage server"""

import duckdb
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager


class DuckDBServer:
    """Manages local DuckDB instance for intermediate storage"""
    
    def __init__(self, db_path: str = "./yuzu_local.duckdb"):
        self.db_path = Path(db_path)
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        
    async def start(self):
        """Initialize DuckDB with schema"""
        self.conn = duckdb.connect(str(self.db_path))
        self._create_schema()
        
    def _create_schema(self):
        """Create local tables mirroring Supabase structure"""
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS seq_session_id START 10000;
            CREATE SEQUENCE IF NOT EXISTS seq_memory_id START 10000;
            
            -- Raw exported data
            CREATE TABLE IF NOT EXISTS messages_export (
                id INTEGER PRIMARY KEY,
                session_id INTEGER,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS sessions_export (
                id INTEGER PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP
            );
            
            -- Processed memories
            CREATE TABLE IF NOT EXISTS episodic_memories_local (
                temp_id INTEGER PRIMARY KEY DEFAULT nextval('seq_memory_id'),
                original_message_id INTEGER,
                session_id INTEGER,
                content TEXT,
                embedding FLOAT[768],
                importance INTEGER DEFAULT 50,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS semantic_memories_local (
                temp_id INTEGER PRIMARY KEY DEFAULT nextval('seq_memory_id'),
                original_message_id INTEGER,
                fact TEXT,
                category TEXT,
                keywords TEXT[],
                embedding FLOAT[768],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS processing_log (
                id INTEGER PRIMARY KEY,
                phase TEXT,
                item_count INTEGER,
                status TEXT,
                message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
    async def stop(self):
        """Close connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            
    def execute(self, sql: str, params: tuple = ()):
        """Execute SQL"""
        return self.conn.execute(sql, params)
        
    def query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Query and return results as dict"""
        result = self.conn.execute(sql, params).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]
        
    def get_stats(self) -> Dict[str, int]:
        """Get current processing stats"""
        try:
            return {
                'messages_count': self.conn.execute("SELECT COUNT(*) FROM messages_export").fetchone()[0],
                'sessions_count': self.conn.execute("SELECT COUNT(*) FROM sessions_export").fetchone()[0],
                'episodic_count': self.conn.execute("SELECT COUNT(*) FROM episodic_memories_local").fetchone()[0],
                'semantic_count': self.conn.execute("SELECT COUNT(*) FROM semantic_memories_local").fetchone()[0],
                'embedded_count': self.conn.execute("SELECT COUNT(*) FROM episodic_memories_local WHERE embedding IS NOT NULL").fetchone()[0],
            }
        except:
            return {}
