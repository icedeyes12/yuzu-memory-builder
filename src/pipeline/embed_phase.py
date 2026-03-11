"""Phase 2: Generate embeddings for messages using local ONNX"""
import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
import numpy as np


@dataclass
class EmbedStats:
    """Embedding statistics"""
    messages_embedded: int = 0
    semantic_facts: int = 0
    errors: int = 0
    cached: int = 0


def extract_facts_from_message(message: str, session_context: str = "") -> List[Dict]:
    """Extract semantic facts from a message for memory layer"""
    facts = []
    message_lower = message.lower()
    
    # Patterns for fact extraction (simplified - can use LLM for better results)
    patterns = [
        # Preferences
        (r'\b(suka|senang|favorit|prefer)\b.*', 'preference'),
        (r'\b(gak suka|benci|jijik)\b.*', 'preference'),
        
        # Identity
        (r'\b(nama (saya|aku)|panggil (saya|aku))\s+(\w+)', 'identity'),
        (r'\b(saya|aku)\s+(adalah|kerja|kuliah)\b', 'identity'),
        
        # Experience
        (r'\b(kemarin|minggu lalu|bulan lalu|tahun lalu)\b.*', 'experience'),
        (r'\b(pernah|sudah)\b.*', 'experience'),
        
        # Goals
        (r'\b(mau|ingin|rencana|target)\b.*', 'goal'),
        (r'\b(akan|bakal)\b.*', 'goal'),
        
        # Relationships
        (r'\b(teman|sahabat|pacar|keluarga|adek|kakak)\b.*', 'relationship'),
    ]
    
    for pattern, category in patterns:
        import re
        matches = re.findall(pattern, message, re.IGNORECASE)
        if matches:
            facts.append({
                'fact': message.strip()[:200],  # Truncate long messages
                'category': category,
                'keywords': extract_keywords(message),
                'source': 'pattern_match',
                'confidence': 0.7
            })
    
    return facts


def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text"""
    # Simple keyword extraction
    import re
    
    # Remove punctuation and split
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter common words
    stopwords = {'dan', 'atau', 'yang', 'dari', 'untuk', 'dengan', 'ini', 'itu', 'saya', 'aku', 'kamu', 'the', 'and', 'for', 'you'}
    keywords = [w for w in words if w not in stopwords]
    
    # Return unique keywords
    return list(set(keywords))[:10]  # Max 10 keywords


def calculate_importance(message: str, role: str = 'user') -> int:
    """Calculate importance score for episodic memory"""
    importance = 50  # Base
    
    # Length bonus
    if len(message) > 100:
        importance += 10
    if len(message) > 300:
        importance += 10
    
    # Content indicators
    indicators_high = ['suka', 'senang', 'favorit', 'important', 'penting', 'ingat', 'remember', 'ga akan lupa']
    indicators_low = ['halo', 'hi', 'hello', 'apa kabar', 'gimana']
    
    msg_lower = message.lower()
    if any(i in msg_lower for i in indicators_high):
        importance += 15
    if any(i in msg_lower for i in indicators_low):
        importance -= 10
    
    # User messages slightly more important
    if role == 'user':
        importance += 5
    
    return max(10, min(100, importance))


class EmbedProcessor:
    """Process messages and generate embeddings"""
    
    def __init__(self, onnx_server, batch_size: int = 32):
        self.onnx = onnx_server
        self.batch_size = batch_size
        
    def process_all(self, messages: List[Dict], progress: Progress) -> Tuple[List[Dict], List[Dict]]:
        """
        Process all messages:
        1. Generate embeddings for episodic memories
        2. Extract semantic facts
        """
        print("\n🔮 Phase 2: Generate Embeddings & Extract Facts\n")
        
        # Create tasks
        embed_task = progress.add_task("[yellow]Embeddings", total=len(messages))
        
        episodic_memories = []
        semantic_facts = []
        
        # Process in batches
        for i in range(0, len(messages), self.batch_size):
            batch = messages[i:i + self.batch_size]
            
            # Generate embeddings
            texts = [m['content'] for m in batch]
            embeddings = self.onnx.embed(texts)
            
            # Create episodic memories
            for msg, embedding in zip(batch, embeddings):
                memory = {
                    'user_id': msg['user_id'],
                    'session_id': msg['session_id'],
                    'content': msg['content'],
                    'embedding': embedding,
                    'importance': calculate_importance(msg['content'], msg.get('role', 'user')),
                    'memory_type': 'episodic',
                    'context': {
                        'people': [],
                        'places': [],
                        'topics': extract_keywords(msg['content']),
                        'sentiment': 'neutral'
                    },
                    'access_count': 0,
                    'surprise': 0.5,
                    'stability': 0,
                    'difficulty': 0,
                    'created_at': msg['created_at']
                }
                episodic_memories.append(memory)
                
                # Extract semantic facts
                if len(msg['content']) > 20:  # Skip very short messages
                    facts = extract_facts_from_message(msg['content'])
                    for fact in facts:
                        semantic_facts.append({
                            'user_id': msg['user_id'],
                            'session_id': msg['session_id'],
                            'fact': fact['fact'],
                            'category': fact['category'],
                            'keywords': fact['keywords'],
                            'embedding': None,  # Will be generated separately
                            'source_memory_id': None,  # Will link after insert
                            'confidence': fact['confidence']
                        })
            
            progress.update(embed_task, advance=len(batch))
            
            if (i // self.batch_size) % 10 == 0:
                print(f"  ✨ Processed {i}/{len(messages)} messages...")
        
        # Generate embeddings for semantic facts
        if semantic_facts:
            print(f"\n📝 Generating embeddings for {len(semantic_facts)} semantic facts...")
            fact_texts = [f['fact'] for f in semantic_facts]
            fact_embeddings = self.onnx.embed(fact_texts)
            
            for fact, embedding in zip(semantic_facts, fact_embeddings):
                fact['embedding'] = embedding
        
        print(f"\n✅ Embedding complete:")
        print(f"   {len(episodic_memories)} episodic memories")
        print(f"   {len(semantic_facts)} semantic facts")
        
        return episodic_memories, semantic_facts


def run_embed_phase(duckdb_server, onnx_server) -> EmbedStats:
    """Run embed phase - main entry point"""
    from rich.progress import Progress
    
    stats = EmbedStats()
    
    # Get messages from DuckDB
    messages = duckdb_server.get_messages()
    if not messages:
        print("❌ No messages to process")
        return stats
    
    print(f"📊 Processing {len(messages)} messages...")
    
    processor = EmbedProcessor(onnx_server)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        episodic, semantic = processor.process_all(messages, progress)
    
    # Store in DuckDB
    duckdb_server.store_episodic_memories(episodic)
    duckdb_server.store_semantic_memories(semantic)
    
    stats.messages_embedded = len(episodic)
    stats.semantic_facts = len(semantic)
    
    return stats
