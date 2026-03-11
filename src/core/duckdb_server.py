"""DuckDB local
[truncated]
"messages
[truncated]
}
        finally:
            self.conn.close()
    def _ensure_tables(self):
        """Create tables if not exist"""
        # ... existing code ...
        
    def store_users(self, users: List[Dict]):
        """Store users in local cache"""
        if not users:
            return
        for user in users:
            self.conn.execute("""
                INSERT OR REPLACE INTO users_export 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user.get('id'), user.get('email'), 
                user.get('display_name'), user.get('partner_name'),
                user.get('auth_token'), user.get('created_at')
            ))
    
    def store_sessions(self, sessions: List[Dict]):
        """Store sessions in local cache"""
        if not sessions:
            return
        for session in sessions:
            self.conn.execute("""
                INSERT OR REPLACE INTO sessions_export
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session.get('id'), session.get('user_id'),
                session.get('title'), session.get('created_at'),
                session.get('updated_at'), session.get('message_count', 0)
            ))
    
    def store_messages(self, messages: List[Dict]):
        """Store messages in local cache"""
        if not messages:
            return
        for msg in messages:
            self.conn.execute("""
                INSERT OR REPLACE INTO messages_export
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                msg.get('id'), msg.get('session_id'), msg.get('user_id'),
                msg.get('role'), msg.get('content'), msg.get('created_at'),
                msg.get('session_title', '')
            ))
    
    def get_messages(self) -> List[Dict]:
        """Get all messages for processing"""
        cur = self.conn.execute("SELECT * FROM messages_export ORDER BY id")
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]
    
    def store_episodic_memories(self, memories: List[Dict]):
        """Store generated episodic memories"""
        if not memories:
            return
        for mem in memories:
            embedding_json = json.dumps(mem.get('embedding', []))
            self.conn.execute("""
                INSERT INTO memories_staging
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                mem.get('user_id'), mem.get('session_id'), mem.get('content'),
                embedding_json, mem.get('importance', 50),
                mem.get('memory_type', 'episodic'), mem.get('created_at')
            ))
    
    def store_semantic_memories(self, memories: List[Dict]):
        """Store generated semantic memories"""
        if not memories:
            return
        for mem in memories:
            embedding_json = json.dumps(mem.get('embedding', [])) if mem.get('embedding') else '[]'
            keywords_json = json.dumps(mem.get('keywords', []))
            self.conn.execute("""
                INSERT INTO semantic_staging
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                mem.get('user_id'), mem.get('session_id'), mem.get('fact'),
                mem.get('category'), keywords_json, embedding_json,
                mem.get('confidence', 0.5)
            ))
    
    def get_memories(self) -> Tuple[List[Dict], List[Dict]]:
        """Get all staged memories for migration"""
        # Get episodic
        cur = self.conn.execute("SELECT * FROM memories_staging")
        columns = [desc[0] for desc in cur.description]
        episodic = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        # Get semantic  
        cur = self.conn.execute("SELECT * FROM semantic_staging")
        columns = [desc[0] for desc in cur.description]
        semantic = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        return episodic, semantic
    
    def close(self):
        if self.conn:
            self.conn.close()
