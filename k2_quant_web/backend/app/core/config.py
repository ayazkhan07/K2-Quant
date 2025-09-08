"""
Application Configuration
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "K2 Quant Web"
    APP_VERSION: str = "8.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # API
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = Field(default="", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql://k2quant:k2quant@localhost/k2quant_web",
        env="DATABASE_URL"
    )
    
    # Redis (for caching and Celery)
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    
    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    
    # External APIs
    POLYGON_API_KEY: str = Field(default="", env="POLYGON_API_KEY")
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = Field(default="", env="ANTHROPIC_API_KEY")
    
    # Stock Data Settings
    MAX_PARALLEL_FETCHES: int = 50
    DEFAULT_CHUNK_SIZE: int = 100000
    
    # WebSocket Settings
    WS_MESSAGE_QUEUE_SIZE: int = 100
    WS_HEARTBEAT_INTERVAL: int = 30
    
    # File Storage
    UPLOAD_DIR: str = Field(default="/tmp/k2quant/uploads", env="UPLOAD_DIR")
    EXPORT_DIR: str = Field(default="/tmp/k2quant/exports", env="EXPORT_DIR")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_DIR: str = Field(default="/tmp/k2quant/logs", env="LOG_DIR")
    
    # Agent System
    ENABLE_AGENTS: bool = Field(default=True, env="ENABLE_AGENTS")
    AGENT_TASK_TIMEOUT: int = 300  # 5 minutes
    
    # Performance
    CACHE_TTL: int = 3600  # 1 hour
    MAX_WORKERS: int = 4
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Create directories if they don't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.EXPORT_DIR, exist_ok=True)
os.makedirs(settings.LOG_DIR, exist_ok=True)