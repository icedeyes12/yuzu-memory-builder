"""Phase 3: Validate embeddings and data quality"""

from typing import Dict, List
from rich.progress import Progress, TaskID
import numpy as np


class ValidatePhase:
    """Validate generated embeddings and data quality"""
    
    def __init__(self, duckdb_server):
        self.duckdb = duckdb_server
        
    def run(self, progress: Progress, task: TaskID) -> Dict:
        """Run validation checks"""
        
        issues = []
        
        # Check 1: All memories have embeddings
        no_embedding = self.duckdb.query("""
            SELECT COUNT(*) as count
            FROM episodic_memories_local
            WHERE embedding IS NULL
        """)
        
        if no_embedding and no_embedding[0]['count'] > 0:
            issues.append(f"{no_embedding[0]['count']} episodic memories missing embeddings")
            
        # Check 2: Embedding dimension
        wrong_dim = self.duckdb.query("""
            SELECT COUNT(*) as count
            FROM episodic_memories_local
            WHERE len(embedding) != 768
        """)
        
        if wrong_dim and wrong_dim[0]['count'] > 0:
            issues.append(f"{wrong_dim[0]['count']} embeddings have wrong dimension")
            
        # Check 3: Duplicate content
        dups = self.duckdb.query("""
            SELECT content, COUNT(*) as cnt
            FROM episodic_memories_local
            GROUP BY content
            HAVING COUNT(*) > 1
            LIMIT 5
        """)
        
        if dups:
            issues.append(f"Found {len(dups)} duplicate contents (showing first 5)")
            
        # Check 4: Sample embeddings (cosine similarity sanity check)
        sample = self.duckdb.query("""
            SELECT content, embedding
            FROM episodic_memories_local
            WHERE embedding IS NOT NULL
            LIMIT 10
        """)
        
        if len(sample) >= 2:
            # Check similar messages have similar embeddings
            emb1 = np.array(sample[0]['embedding'])
            emb2 = np.array(sample[1]['embedding'])
            
            # Cosine similarity
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            if sim > 0.99:  # Too similar = possible bug
                issues.append("Warning: Sample embeddings too similar (possible bug)")
            elif sim < -0.5:  # Too different = possible bug
                issues.append("Warning: Sample embeddings too different (possible bug)")
                
        progress.update(task, completed=100)
        
        # Stats
        stats = self.duckdb.get_stats()
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'stats': stats
        }
