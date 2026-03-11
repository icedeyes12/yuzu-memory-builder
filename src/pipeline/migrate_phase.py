"""Phase 4: Migrate from DuckDB to Supabase"""

from typing import Dict
from rich.progress import Progress, TaskID


class MigratePhase:
    """Safe migration to Supabase"""
    
    def __init__(self, duckdb_server, supabase_client):
        self.duckdb = duckdb_server
        self.supabase = supabase_client
        
    def run(self, progress: Progress, task: TaskID, 
            user_email: str, display_name: str, 
            partner_name: str, auth_token: str) -> Dict[str, int]:
        """Migrate all data to Supabase"""
        
        # Step 1: Get or create user
        user = self.supabase.get_existing_user(user_email)
        if user:
            user_id = user['id']
            print(f"⚠️  User exists (id={user_id}), will append memories")
        else:
            user_id = self.supabase.create_migrated_user(
                user_email, display_name, partner_name, auth_token
            )
            print(f"✅ Created user (id={user_id})")
            
        progress.update(task, completed=10)
        
        # Step 2: Migrate episodic memories
        episodic = self.duckdb.query("""
            SELECT original_message_id, session_id, content, 
                   embedding, importance, created_at
            FROM episodic_memories_local
            WHERE embedding IS NOT NULL
        """)
        
        episodic_migrated = self.supabase.migrate_episodic_memories(
            episodic, user_id
        )
        progress.update(task, completed=60)
        
        # Step 3: Migrate semantic memories
        semantic = self.duckdb.query("""
            SELECT original_message_id, fact, category, 
                   keywords, embedding, created_at
            FROM semantic_memories_local
            WHERE embedding IS NOT NULL
        """)
        
        semantic_migrated = self.supabase.migrate_semantic_memories(
            semantic, user_id
        )
        progress.update(task, completed=100)
        
        # Get final stats
        supabase_stats = self.supabase.get_stats()
        
        return {
            'user_id': user_id,
            'episodic_migrated': episodic_migrated,
            'semantic_migrated': semantic_migrated,
            'episodic_total': supabase_stats['episodic_count'],
            'semantic_total': supabase_stats['semantic_count']
        }
