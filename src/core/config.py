"""Configuration management"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Config(BaseSettings):
    """Application configuration"""
    
    # Supabase (required)
    database_url: str = Field(default="", alias="DATABASE_URL")
    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    supabase_key: str = Field(default="", alias="SUPABASE_KEY")
    
    # JWT (optional)
    jwt_secret: str = Field(default="", alias="JWT_SECRET")
    secret_key: str = Field(default="", alias="SECRET_KEY")
    
    # HuggingFace (optional - for model download)
    hf_token: str = Field(default="", alias="HF_TOKEN")
    
    # Local
    local_db_path: str = Field(default="./yuzu_local.duckdb", alias="LOCAL_DB_PATH")
    
    # Model
    model_name: str = Field(default="intfloat/multilingual-e5-base", alias="MODEL_NAME")
    onnx_threads: int = Field(default=4, alias="ONNX_THREADS")
    e5_batch_size: int = Field(default=768, alias="E5_BATCH_SIZE")
    
    # Processing
    batch_size: int = Field(default=50, alias="BATCH_SIZE")
    max_workers: int = Field(default=2, alias="MAX_WORKERS")
    enable_semantic: bool = Field(default=True, alias="ENABLE_SEMANTIC")
    enable_fsrs: bool = Field(default=True, alias="ENABLE_FSRS")
    enable_embed: bool = Field(default=True, alias="ENABLE_EMBED")
    
    # Safety
    dry_run: bool = Field(default=True, alias="DRY_RUN")
    validate_before_migrate: bool = Field(default=True, alias="VALIDATE_BEFORE_MIGRATE")
    skip_migrate_on_error: bool = Field(default=False, alias="SKIP_MIGRATE_ON_ERROR")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"
        
    @property
    def model_path(self) -> Path:
        """Path to downloaded ONNX model"""
        return Path("./models") / self.model_name.replace("/", "--")
