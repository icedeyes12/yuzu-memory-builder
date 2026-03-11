"""Configuration management"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Config(BaseSettings):
    """Application configuration"""
    
    # Supabase
    database_url: str = Field(..., alias="DATABASE_URL")
    jwt_secret: str = Field(..., alias="JWT_SECRET")
    secret_key: str = Field(..., alias="SECRET_KEY")
    
    # Local
    local_db_path: str = Field(default="./yuzu_local.duckdb", alias="LOCAL_DB_PATH")
    
    # Model
    model_name: str = Field(default="intfloat/multilingual-e5-large", alias="MODEL_NAME")
    
    # Processing
    batch_size: int = Field(default=50, alias="BATCH_SIZE")
    max_workers: int = Field(default=2, alias="MAX_WORKERS")
    enable_semantic: bool = Field(default=True, alias="ENABLE_SEMANTIC")
    enable_fsrs: bool = Field(default=True, alias="ENABLE_FSRS")
    
    # Safety
    dry_run: bool = Field(default=True, alias="DRY_RUN")
    validate_before_migrate: bool = Field(default=True, alias="VALIDATE_BEFORE_MIGRATE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
    @property
    def model_path(self) -> Path:
        """Path to downloaded ONNX model"""
        return Path("./models") / self.model_name.replace("/", "--")
