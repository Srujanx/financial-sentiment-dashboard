import os
import logging
from pathlib import Path
from functools import lru_cache

class Settings:
    # API Configuration
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    ALPHA_VANTAGE_BASE_URL: str = "https://www.alphavantage.co/query"
    
    # Application Settings
    APP_NAME: str = "Financial Sentiment Dashboard"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Model Configuration
    SENTIMENT_MODEL: str = "ProsusAI/finbert"
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", str(Path.home() / ".cache" / "huggingface"))
    
    # Cache Configuration
    CACHE_TTL: int = 3600  # 1 hour in seconds
    MAX_CACHE_SIZE: int = 100  # Maximum number of cached items
    
    # API Rate Limiting
    RATE_LIMIT_CALLS: int = 25  # Alpha Vantage free tier limit
    RATE_LIMIT_PERIOD: int = 86400  # 24 hours in seconds
    
    # Sentiment Analysis
    MIN_RELEVANCE_SCORE: float = 0.4  # Minimum relevance for news filtering
    MIN_CONFIDENCE_SCORE: float = 0.6  # Minimum confidence for high-quality predictions
    MAX_TEXT_LENGTH: int = 512  # Maximum tokens for model input
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # FastAPI Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))
    
    # CORS
    ALLOWED_ORIGINS: list = [
        "http://localhost:8501",  # Streamlit default
        "http://frontend:8501",   # Docker service name
    ]
  # Ticker Configuration
    DEFAULT_TICKER: str = "AAPL"
    ALLOWED_TICKERS: list = ["AAPL"]  

    def validate(self):  # ← Make sure it has 'self' parameter
        """Validate configuration"""
        if not self.ALPHA_VANTAGE_API_KEY:
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable is required")
        
        if self.DEFAULT_TICKER not in self.ALLOWED_TICKERS:
            raise ValueError(f"DEFAULT_TICKER must be one of {self.ALLOWED_TICKERS}")
        
        return True
@lru_cache
def get_settings() -> Settings:
    return Settings

def setup_logging():
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=settings.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("/app/logs/app.log")
        ]
    )
# Set third-party library log levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


# Initialize logger
logger = setup_logging()

