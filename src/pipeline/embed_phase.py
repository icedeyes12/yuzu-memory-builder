"""Phase 2: Generate embeddings for episodic and semantic memories"""

from typing import Dict, List
from rich.progress import Progress, TaskID
import re


class EmbedPhase:
    """Generate embeddings using ONNX model"""
    
    def __init__(self, duckdb_server, onnx_server):
        self.duckdb = duckdb_server
        self.onnx = onnx_server
        
    def run(self, progress: Progress, task: TaskID) -> Dict[str, int]:
        """Process all messages and generate embeddings"""
        
        # Get unprocessed messages
        messages = self.duckdb.query("""
            SELECT id, session_id, role, content, created_at
            FROM messages_export
            WHERE id NOT IN (SELECT original_message_id FROM episodic_memories_local)
            ORDER BY id
        """)
        
        if not messages:
            return {'episodic': 0, 'semantic': 0}
            
        total = len(messages)
        processed = 0
        episodic_count = 0
        semantic_count = 0
        
        # Process in batches of 32 (memory efficient)
        batch_size = 32
        
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            
            # Generate episodic memories (all user messages)
            episodic = self._create_episodic(batch)
            if episodic:
                self._save_episodic(episodic)
                episodic_count += len(episodic)
                
            # Generate semantic memories (extract facts)
            semantic = self._extract_semantic(batch)
            if semantic:
                embeddings = self.onnx.embed([s['fact'] for s in semantic])
                for s, emb in zip(semantic, embeddings):
                    s['embedding'] = emb
                self._save_semantic(semantic)
                semantic_count += len(semantic)
                
            processed += len(batch)
            progress.update(task, completed=processed, total=total)
            
        return {
            'episodic': episodic_count,
            'semantic': semantic_count
        }
        
    def _create_episodic(self, messages: List[Dict]) -> List[Dict]:
        """Create episodic memories from messages"""
        memories = []
        
        for msg in messages:
            # Skip very short messages
            if len(msg['content']) < 10:
                continue
                
            # Calculate importance (simple heuristic)
            importance = 50
            if msg['role'] == 'user':
                importance += 10
            if len(msg['content']) > 100:
                importance += 10
                
            memories.append({
                'original_message_id': msg['id'],
                'session_id': msg['session_id'],
                'content': msg['content'][:500],  # Truncate very long
                'importance': min(importance, 90)
            })
            
        return memories
        
    def _extract_semantic(self, messages: List[Dict]) -> List[Dict]:
        """Extract semantic facts from messages"""
        facts = []
        
        for msg in messages:
            content = msg['content']
            
            # Pattern 1: "I like...", "I love...", "My favorite..."
            patterns = [
                (r'\b(?:suka|like|love|enjoy)\s+(.{10,100})', 'preference'),
                (r'\b(?:tidak suka|hate|dislike)\s+(.{10,100})', 'preference'),
                (r'\bfavorite\s+(.{10,50})', 'preference'),
                (r'\b(?:kerja di|work at|job)\s+(.{10,80})', 'identity'),
                (r'\b(?:tinggal di|live in|dari)\s+(.{10,50})', 'identity'),
                (r'\b(?:hobi|hobby)\s+(.{10,100})', 'interest'),
            ]
            
            for pattern, category in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    fact_text = match.group(1).strip()
                    if len(fact_text) > 10:
                        facts.append({
                            'original_message_id': msg['id'],
                            'fact': fact_text[:200],
                            'category': category,
                            'keywords': self._extract_keywords(fact_text)
                        })
                        
        return facts
        
    def _extract_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction"""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'di', 'yang', 'dan'}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]
        return list(set(keywords))[:5]  # Max 5 keywords
        
    def _save_episodic(self, memories: List[Dict]):
        """Save to DuckDB"""
        if not memories:
            return
            
        # Get embeddings
        texts = [m['content'] for m in memories]
        embeddings = self.onnx.embed(texts)
        
        # Insert with embeddings
        self.duckdb.execute("BEGIN TRANSACTION")
        for m, emb in zip(memories, embeddings):
            self.duckdb.execute("""
                INSERT INTO episodic_memories_local 
                    (original_message_id, session_id, content, embedding, importance)
                VALUES (?, ?, ?, ?, ?)
            """, (
                m['original_message_id'],
                m['session_id'],
                m['content'],
                emb,
                m['importance']
            ))
        self.duckdb.execute("COMMIT")
        
    def _save_semantic(self, facts: List[Dict]):
        """Save semantic to DuckDB"""
        if not facts:
            return
            
        self.duckdb.execute("BEGIN TRANSACTION")
        for f in facts:
            self.duckdb.execute("""
                INSERT INTO semantic_memories_local
                    (original_message_id, fact, category, keywords, embedding)
                VALUES (?, ?, ?, ?, ?)
            """, (
                f['original_message_id'],
                f['fact'],
                f['category'],
                f['keywords'],
                f.get('embedding')
            ))
        self.duckdb.execute("COMMIT")
