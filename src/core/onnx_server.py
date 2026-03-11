"""ONNX embedding server with memory-efficient loading"""

import os
import asyncio
from pathlib import Path
from typing import List, Optional
import numpy as np


class ONNXServer:
    """Singleton ONNX runtime for embeddings"""
    
    _instance = None
    
    def __new__(cls, model_name: str = "intfloat/multilingual-e5-large"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        if self._initialized:
            return
            
        self.model_name = model_name
        self.model_path = Path("./models") / model_name.replace("/", "--")
        self.session = None
        self.tokenizer = None
        self._initialized = True
        self._lock = asyncio.Lock()
        
    def download_model(self):
        """Download model from HuggingFace"""
        if self.model_path.exists():
            return
            
        from huggingface_hub import snapshot_download
        
        print(f"📥 Downloading {self.model_name}...")
        snapshot_download(
            repo_id=self.model_name,
            local_dir=str(self.model_path),
            local_dir_use_symlinks=False,
            allow_patterns=["*.onnx", "*.json", "*.txt"]
        )
        print(f"✅ Model downloaded to {self.model_path}")
        
    async def start(self):
        """Lazy load - model loads on first embed() call"""
        pass  # Lazy loading
        
    async def stop(self):
        """Cleanup"""
        self.session = None
        self.tokenizer = None
        ONNXServer._instance = None
        
    def _load(self):
        """Load model if not already loaded"""
        if self.session is not None:
            return
            
        import onnxruntime as ort
        from tokenizers import Tokenizer
        
        # Find ONNX file
        onnx_files = list(self.model_path.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX file found in {self.model_path}")
            
        model_file = onnx_files[0]
        
        # Load with memory-efficient settings
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        
        self.session = ort.InferenceSession(
            str(model_file),
            sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Load tokenizer
        tokenizer_file = self.model_path / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        self._load()
        
        # Tokenize
        encoded = self.tokenizer.encode_batch(texts)
        
        # Prepare inputs
        max_len = max(len(e.ids) for e in encoded)
        input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)
        
        for i, e in enumerate(encoded):
            input_ids[i, :len(e.ids)] = e.ids
            attention_mask[i, :len(e.ids)] = [1] * len(e.ids)
            
        # Run inference
        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        )
        
        # Mean pooling
        embeddings = []
        for i, output in enumerate(outputs[0]):
            mask = attention_mask[i]
            embedding = output[:len([m for m in mask if m])].mean(axis=0)
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embeddings.append(embedding.tolist())
            
        return embeddings
        
    def embed_single(self, text: str) -> List[float]:
        """Embed single text"""
        return self.embed([text])[0]
