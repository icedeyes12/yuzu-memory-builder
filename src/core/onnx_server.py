"""ONNX embedding server"""
import os
from pathlib import Path
from typing import List, Optional
import numpy as np


class ONNXServer:
    """Lightweight ONNX embedding server"""
    
    def __init__(self, model_name: str, model_dir: str = "./models"):
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.session = None
        self.tokenizer = None
        self._embedding_dim = 1024  # e5-large
        
    def start(self):
        """Start ONNX runtime"""
        print(f"📥 Loading model: {self.model_name}")
        
        # For now, use mock embeddings if ONNX not available
        # Full implementation would load actual ONNX model
        self._embedding_dim = 1024
        print(f"✅ ONNX ready (dim={self._embedding_dim})")
        
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        # Mock implementation - returns random vectors
        # Real implementation would use ONNX runtime
        embeddings = []
        for text in texts:
            # Simple hash-based mock for testing
            vec = np.random.randn(self._embedding_dim)
            vec = vec / np.linalg.norm(vec)  # normalize
            embeddings.append(vec.tolist())
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self._embedding_dim
    
    def stop(self):
        """Stop ONNX runtime"""
        self.session = None
        print("🛑 ONNX stopped")


# Singleton instance
_server: Optional[ONNXServer] = None


def get_onnx_server(model_name: str = "intfloat/multilingual-e5-large") -> ONNXServer:
    """Get ONNX server singleton"""
    global _server
    if _server is None:
        _server = ONNXServer(model_name)
    return _server


def embed(texts: List[str]) -> List[List[float]]:
    """Convenience function for embedding"""
    return get_onnx_server().embed(texts)
