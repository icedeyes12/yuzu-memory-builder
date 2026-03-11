"""ONNX embedding server - runs in background for embedding generation"""
import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
import numpy as np

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
except ImportError:
    print("❌ ONNX Runtime not installed. Run: pip install onnxruntime tokenizers")
    sys.exit(1)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model"""
    model_name: str = "intfloat/multilingual-e5-large"
    max_seq_length: int = 512
    batch_size: int = 32
    normalize: bool = True
    
    @property
    def model_dir(self) -> Path:
        return Path("./models") / self.model_name.replace("/", "--")
    
    @property
    def onnx_path(self) -> Path:
        return self.model_dir / "onnx/model.onnx"
    
    @property
    def tokenizer_path(self) -> Path:
        return self.model_dir / "tokenizer.json"


class ONNXServer:
    """Singleton ONNX server for embedding generation"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        if self._initialized:
            return
        
        self.config = config or EmbeddingConfig()
        self.session = None
        self.tokenizer = None
        self.embedding_dim = 1024  # e5-large dimension
        self._initialized = False
    
    def start(self) -> bool:
        """Start the ONNX server - load model"""
        if self._initialized:
            return True
        
        try:
            # Check if model exists
            if not self.config.onnx_path.exists():
                print(f"❌ Model not found: {self.config.onnx_path}")
                print("Please download model first with: python3 -c \"from main import download_model; download_model()\"")
                return False
            
            # Load tokenizer
            self.tokenizer = Tokenizer.from_file(str(self.config.tokenizer_path))
            self.tokenizer.enable_truncation(max_length=self.config.max_seq_length)
            self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            
            # Load ONNX model
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 4
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                str(self.config.onnx_path),
                sess_options,
                providers=['CPUExecutionProvider']
            )
            
            self._initialized = True
            print(f"✅ ONNX Server started - {self.config.model_name} ({self.embedding_dim}D)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start ONNX server: {e}")
            return False
    
    def stop(self):
        """Stop the ONNX server"""
        self.session = None
        self.tokenizer = None
        self._initialized = False
        print("✅ ONNX Server stopped")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        if not self._initialized:
            raise RuntimeError("ONNX server not started")
        
        if not texts:
            return []
        
        results = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_results = self._embed_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts"""
        # Add "passage: " prefix for e5 models
        prefixed = [f"passage: {t}" for t in texts]
        
        # Tokenize
        encoded = self.tokenizer.encode_batch(prefixed)
        
        # Prepare inputs
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        
        # Run inference
        outputs = self.session.run(
            None,
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        )
        
        # Mean pooling
        last_hidden = outputs[0]  # [batch, seq_len, hidden_dim]
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = (last_hidden * mask_expanded).sum(axis=1)
        embeddings = sum_embeddings / mask_expanded.sum(axis=1)
        
        # Normalize
        if self.config.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-12)
        
        return embeddings.tolist()
    
    def embed_single(self, text: str) -> List[float]:
        """Embed single text"""
        return self.embed([text])[0] if text else []


# Global singleton accessor
def get_onnx_server(config: Optional[EmbeddingConfig] = None) -> ONNXServer:
    """Get or create ONNX server singleton"""
    return ONNXServer(config)


def download_model(model_name: str = "intfloat/multilingual-e5-large", 
                   token: Optional[str] = None) -> bool:
    """Download ONNX model from HuggingFace"""
    from huggingface_hub import snapshot_download
    
    config = EmbeddingConfig(model_name=model_name)
    
    try:
        print(f"⬇️  Downloading {model_name}...")
        
        snapshot_download(
            repo_id=model_name,
            local_dir=str(config.model_dir),
            local_dir_use_symlinks=False,
            token=token,
            ignore_patterns=["*.bin", "*.safetensors", "*.pt", "*.pth", "*.msgpack"]
        )
        
        print(f"✅ Model downloaded to {config.model_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print("Make sure you have HF_TOKEN set for gated models")
        return False


if __name__ == "__main__":
    # Test server
    server = get_onnx_server()
    if server.start():
        test = server.embed_single("Hello world")
        print(f"Test embedding: {len(test)} dimensions")
        server.stop()
