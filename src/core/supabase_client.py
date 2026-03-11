"""Supabase client for safe migration"""

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
from contextlib import contextmanager


class SupabaseClient:
    """Safe Supabase operations with batching and validation"""
    
    def __init__(self, connection_string: str):
        self.conn_string = connection_string
        self.conn = None
        self._batch_size = 100  # Insert 100 rows at a time
        
    def connect(self):
        """Connect to Supabase"""
        self.conn = psycopg2.connect(self.conn_string)
        print("✅ Connected to Supabase")
        
    def disconnect(self):
        """Close connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            
    def validate_schema(self) -> bool:
        """Check required tables exist"""
        required_tables = [
            'users', 'chat_sessions', 'messages',
            'memories', 'semantic_memories'
        ]
        
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            existing = {row[0] for row in cur.fetchall()}
            
        missing = set(required_tables) - existing
        if missing:
            print(f"❌ Missing tables: {missing}")
            return False
            
        print("✅ Schema validated")
        return True
        
    def get_existing_user(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            result = cur.fetchone()
            return dict(result) if result else None
            
    def create_migrated_user(self, email: str, display_name: str, 
                             partner_name: str, auth_token: str) -> int:
        """Create user for migrated data"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (email, password_hash, auth_token, 
                                   display_name, partner_name, affection)
                VALUES (%s, 'MIGRATED', %s, %s, %s, 85)
                ON CONFLICT (email) DO UPDATE SET
                    auth_token = EXCLUDED.auth_token,
                    updated_at = NOW()
                RETURNING id
            """, (email, auth_token, display_name, partner_name))
            
            user_id = cur.fetchone()[0]
            self.conn.commit()
            return user_id
            
    def migrate_episodic_memories(self, memories: List[Dict], user_id: int) -> int:
        """Batch insert episodic memories"""
        if not memories:
            return 0
            
        inserted = 0
        
        with self.conn.cursor() as cur:
            for batch in self._batch(memories, self._batch_size):
                args = []
                for m in batch:
                    args.extend([
                        user_id,
                        m.get('session_id'),
                        m.get('content'),
                        m.get('embedding'),
                        m.get('importance', 50)
                    ])
                    
                # Build VALUES clause
                values = ','.join(['(%s,%s,%s,%s::vector(768),%s)'] * len(batch))
                
                cur.execute(f"""
                    INSERT INTO memories 
                        (user_id, session_id, content, embedding, importance, memory_type)
                    VALUES {values}
                    ON CONFLICT DO NOTHING
                """, args)
                
                inserted += cur.rowcount
                
            self.conn.commit()
            
        return inserted
        
    def migrate_semantic_memories(self, memories: List[Dict], user_id: int) -> int:
        """Batch insert semantic memories"""
        if not memories:
            return 0
            
        inserted = 0
        
        with self.conn.cursor() as cur:
            for batch in self._batch(memories, self._batch_size):
                args = []
                for m in batch:
                    args.extend([
                        user_id,
                        m.get('fact'),
                        m.get('category', 'identity'),
                        m.get('keywords', []),
                        m.get('embedding')
                    ])
                    
                values = ','.join(['(%s,%s,%s,%s,%s::vector(768))'] * len(batch))
                
                cur.execute(f"""
                    INSERT INTO semantic_memories 
                        (user_id, fact, category, keywords, embedding)
                    VALUES {values}
                    ON CONFLICT DO NOTHING
                """, args)
                
                inserted += cur.rowcount
                
            self.conn.commit()
            
        return inserted
        
    def get_stats(self) -> Dict[str, int]:
        """Get migration stats"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM memories")
            episodic = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM semantic_memories")
            semantic = cur.fetchone()[0]
            
            return {
                'episodic_count': episodic,
                'semantic_count': semantic
            }
            
    @staticmethod
    def _batch(items: List[Any], size: int):
        """Split list into batches"""
        for i in range(0, len(items), size):
            yield items[i:i + size]
