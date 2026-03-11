"""Phase 1: Export data from Supabase to local DuckDB"""
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
import psycopg2
from psycopg2.extras import RealDictCursor


@dataclass
class ExportStats:
    """Export statistics"""
    sessions_exported: int = 0
    messages_exported: int = 0
    users_exported: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class SupabaseExporter:
    """Export data from Supabase PostgreSQL"""
    
    def __init__(self, connection_string: str, batch_size: int = 1000):
        self.conn_string = connection_string
        self.batch_size = batch_size
        self.conn: Optional[psycopg2.extensions.connection] = None
        
    def connect(self) -> bool:
        """Connect to Supabase"""
        try:
            self.conn = psycopg2.connect(self.conn_string)
            print("✅ Connected to Supabase")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Close connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def get_stats(self) -> Dict[str, int]:
        """Get database stats"""
        if not self.conn:
            return {}
        
        with self.conn.cursor() as cur:
            stats = {}
            
            # Count tables
            for table in ['users', 'chat_sessions', 'messages', 'memories']:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[table] = cur.fetchone()[0]
                except:
                    stats[table] = 0
            
            return stats
    
    def export_users(self, progress: Progress, task: TaskID) -> List[Dict]:
        """Export all users"""
        if not self.conn:
            return []
        
        users = []
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users ORDER BY id")
            
            while True:
                rows = cur.fetchmany(self.batch_size)
                if not rows:
                    break
                users.extend([dict(row) for row in rows])
                progress.update(task, advance=len(rows))
        
        return users
    
    def export_sessions(self, progress: Progress, task: TaskID) -> List[Dict]:
        """Export all chat sessions with message counts"""
        if not self.conn:
            return []
        
        sessions = []
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    cs.id, 
                    cs.user_id, 
                    cs.title, 
                    cs.created_at, 
                    cs.updated_at,
                    COUNT(m.id) as message_count
                FROM chat_sessions cs
                LEFT JOIN messages m ON m.session_id = cs.id
                GROUP BY cs.id, cs.user_id, cs.title, cs.created_at, cs.updated_at
                ORDER BY cs.id
            """)
            
            while True:
                rows = cur.fetchmany(self.batch_size)
                if not rows:
                    break
                sessions.extend([dict(row) for row in rows])
                progress.update(task, advance=len(rows))
        
        return sessions
    
    def export_messages(self, progress: Progress, task: TaskID) -> List[Dict]:
        """Export all messages - PAGINATED for 7k+ messages"""
        if not self.conn:
            return []
        
        messages = []
        last_id = 0
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            while True:
                cur.execute("""
                    SELECT 
                        m.id, 
                        m.session_id, 
                        m.user_id,
                        m.role, 
                        m.content, 
                        m.created_at,
                        cs.title as session_title
                    FROM messages m
                    JOIN chat_sessions cs ON cs.id = m.session_id
                    WHERE m.id > %s
                    ORDER BY m.id
                    LIMIT %s
                """, (last_id, self.batch_size))
                
                rows = cur.fetchall()
                if not rows:
                    break
                
                batch = [dict(row) for row in rows]
                messages.extend(batch)
                last_id = batch[-1]['id']
                
                progress.update(task, advance=len(batch))
                
                # Print progress every 1000 messages
                if len(messages) % 1000 == 0:
                    print(f"  📦 Exported {len(messages)} messages...")
        
        return messages
    
    def export_all(self, progress: Progress) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Export all data: users, sessions, messages"""
        print("\n📤 Phase 1: Export from Supabase")
        
        # Get stats first
        stats = self.get_stats()
        total_messages = stats.get('messages', 0)
        print(f"   Found: {stats.get('users', 0)} users, {stats.get('chat_sessions', 0)} sessions, {total_messages} messages\n")
        
        # Create progress tasks
        user_task = progress.add_task("[cyan]Users", total=stats.get('users', 1))
        session_task = progress.add_task("[cyan]Sessions", total=stats.get('chat_sessions', 1))
        message_task = progress.add_task("[cyan]Messages", total=total_messages or None)
        
        # Export
        users = self.export_users(progress, user_task)
        sessions = self.export_sessions(progress, session_task)
        messages = self.export_messages(progress, message_task)
        
        print(f"\n✅ Export complete:")
        print(f"   {len(users)} users")
        print(f"   {len(sessions)} sessions")
        print(f"   {len(messages)} messages")
        
        return users, sessions, messages


def run_export_phase(conn_string: str, duckdb_server) -> ExportStats:
    """Run export phase - main entry point"""
    exporter = SupabaseExporter(conn_string)
    stats = ExportStats()
    
    if not exporter.connect():
        stats.errors.append("Failed to connect to Supabase")
        return stats
    
    try:
        from rich.progress import Progress
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})")
        ) as progress:
            users, sessions, messages = exporter.export_all(progress)
        
        # Store in DuckDB
        duckdb_server.store_users(users)
        duckdb_server.store_sessions(sessions)
        duckdb_server.store_messages(messages)
        
        stats.users_exported = len(users)
        stats.sessions_exported = len(sessions)
        stats.messages_exported = len(messages)
        
    except Exception as e:
        stats.errors.append(str(e))
        print(f"❌ Export error: {e}")
    
    finally:
        exporter.disconnect()
    
    return stats
