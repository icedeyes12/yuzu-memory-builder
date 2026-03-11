"""Supabase client for migration"""
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional


class SupabaseClient:
    """Safe Supabase operations"""
    
    def __init__(self, connection_string: str):
        self.conn_string = connection_string
        self.conn = None
        
    def connect(self):
        """Connect to Supabase"""
        self.conn = psycopg2.connect(self.conn_string)
        print("✅ Connected to Supabase")
        
    def disconnect(self):
        """Close connection"""
        if self.conn:
            self.conn.close()
            
    def fetch_messages(self, limit: int = 1000, offset: int = 0) -> List[Dict]:
        """Fetch messages from Supabase"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT m.id, m.session_id, m.role, m.content, m.created_at,
                       cs.title as session_title
                FROM messages m
                JOIN chat_sessions cs ON m.session_id = cs.id
                ORDER BY m.created_at
                LIMIT %s OFFSET %s
            """, (limit, offset))
            return [dict(row) for row in cur.fetchall()]
            
    def fetch_sessions(self) -> List[Dict]:
        """Fetch all sessions"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id, title, user_id, created_at FROM chat_sessions")
            return [dict(row) for row in cur.fetchall()]
            
    def get_user(self, email: str = None, user_id: int = None) -> Optional[Dict]:
        """Get user by email or ID"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            if email:
                cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            elif user_id:
                cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            else:
                return None
            result = cur.fetchone()
            return dict(result) if result else None
            
    def create_user(self, email: str, display_name: str, partner_name: str, auth_token: str) -> int:
        """Create migrated user"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (email, password_hash, auth_token, 
                                  display_name, partner_name, affection)
                VALUES (%s, 'MIGRATED', %s, %s, %s, 85)
                ON CONFLICT (email) DO UPDATE SET auth_token = EXCLUDED.auth_token
                RETURNING id
            """, (email, auth_token, display_name, partner_name))
            user_id = cur.fetchone()[0]
            self.conn.commit()
            return user_id
            
    def migrate_memories(self, memories: List[Dict]) -> int:
        """Batch insert memories"""
        if not memories:
            return 0
            
        inserted = 0
        batch_size = 50
        
        with self.conn.cursor() as cur:
            for i in range(0, len(memories), batch_size):
                batch = memories[i:i + batch_size]
                args = []
                for m in batch:
                    emb = m.get('embedding')
                    emb_str = f"[{','.join(map(str, emb))}]" if emb else None
                    args.extend([
                        m.get('user_id'),
                        m.get('session_id'),
                        m.get('content'),
                        emb_str,
                        m.get('importance', 50),
                        m.get('memory_type', 'episodic')
                    ])
                    
                values = ','.join(['(%s,%s,%s,%s::vector(1024),%s,%s)'] * len(batch))
                cur.execute(f"""
                    INSERT INTO memories 
                        (user_id, session_id, content, embedding, importance, memory_type)
                    VALUES {values}
                    ON CONFLICT DO NOTHING
                """, args)
                inserted += cur.rowcount
                
            self.conn.commit()
        return inserted
        
    def get_stats(self) -> Dict[str, int]:
        """Get Supabase stats"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM memories")
            memories = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM semantic_memories")
            semantic = cur.fetchone()[0]
            return {'memories': memories, 'semantic': semantic}
