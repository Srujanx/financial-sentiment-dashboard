import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from functools import lru_cache
import time
import re

from app.config import get_settings
from app.models import NewsItem

logger = logging.getLogger(__name__)
settings = get_settings()

# Cache for API responses
_cache = {}
_cache_timestamps = {}


def _clear_cache():
    """Clear the cache (for testing)"""
    global _cache, _cache_timestamps
    _cache.clear()
    _cache_timestamps.clear()


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse Alpha Vantage timestamp format (YYYYMMDDTHHMISS)
    
    Args:
        timestamp_str: Timestamp string from Alpha Vantage
        
    Returns:
        datetime object
    """
    try:
        return datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")
    except ValueError as e:
        logger.warning(f"Failed to parse timestamp {timestamp_str}: {e}")
        return datetime.now()


def validate_ticker(ticker: str) -> bool:
    """
    Validate ticker symbol format
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If ticker is invalid
    """
    if not ticker:
        raise ValueError("Ticker cannot be empty")
    
    # Check for valid ticker format (letters only, 1-5 characters)
    if not re.match(r'^[A-Z]{1,5}$', ticker.upper()):
        raise ValueError(f"Invalid ticker symbol: {ticker}")
    
    # Check against allowed tickers
    if ticker.upper() not in settings.ALLOWED_TICKERS:
        raise ValueError(f"Ticker {ticker} is not in allowed list: {settings.ALLOWED_TICKERS}")
    
    return True


def _get_cache_key(ticker: str) -> str:
    """Generate cache key for ticker"""
    return f"news_{ticker.upper()}"


def _is_cache_valid(cache_key: str) -> bool:
    """Check if cached data is still valid"""
    if cache_key not in _cache_timestamps:
        return False
    
    cache_time = _cache_timestamps[cache_key]
    age = time.time() - cache_time
    
    return age < settings.CACHE_TTL


def fetch_financial_news(
    api_key: str,
    ticker: str = "AAPL",
    min_relevance: float = None
) -> List[Dict]:
    """
    Fetch financial news from Alpha Vantage API
    
    Args:
        api_key: Alpha Vantage API key
        ticker: Stock ticker symbol
        min_relevance: Minimum relevance score filter (default from settings)
        
    Returns:
        List of news items
        
    Raises:
        ValueError: If API returns an error or rate limit exceeded
        ConnectionError: If network request fails
        TimeoutError: If request times out
    """
    # Validate ticker
    validate_ticker(ticker)
    
    # Check cache first
    cache_key = _get_cache_key(ticker)
    if _is_cache_valid(cache_key):
        logger.info(f"Returning cached news for {ticker}")
        return _cache[cache_key]
    
    # Set default min_relevance
    if min_relevance is None:
        min_relevance = settings.MIN_RELEVANCE_SCORE
    
    logger.info(f"Fetching news for {ticker} from Alpha Vantage API")
    
    # Prepare API request
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker.upper(),
        "apikey": api_key,
        "limit": 50  # Get up to 50 news items
    }
    
    try:
        response = requests.get(
            settings.ALPHA_VANTAGE_BASE_URL,
            params=params,
            timeout=10
        )
        
        # Check for network errors
        if response.status_code != 200:
            logger.error(f"API returned status code {response.status_code}")
            raise ValueError(f"API Error: HTTP {response.status_code}")
        
        data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            logger.error(f"API Error: {data['Error Message']}")
            raise ValueError(f"API Error: {data['Error Message']}")
        
        # Check for rate limit
        if "Note" in data:
            logger.error(f"Rate limit exceeded: {data['Note']}")
            raise ValueError(f"Rate limit exceeded. Please try again later.")
        
        # Extract news feed
        feed = data.get("feed", [])
        
        if not feed:
            logger.warning(f"No news found for {ticker}")
            return []
        
        # Process news items
        news_items = []
        for item in feed:
            try:
                # Find ticker-specific sentiment
                ticker_sentiment = None
                for ts in item.get("ticker_sentiment", []):
                    if ts.get("ticker") == ticker.upper():
                        ticker_sentiment = ts
                        break
                
                if not ticker_sentiment:
                    continue
                
                # Filter by relevance
                relevance = float(ticker_sentiment.get("relevance_score", 0))
                if relevance < min_relevance:
                    continue
                
                # Parse news item
                news_item = {
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", "Unknown"),
                    "time_published": parse_timestamp(item.get("time_published", "")),
                    "ticker": ticker.upper(),
                    "relevance_score": relevance,
                    "sentiment_score": float(ticker_sentiment.get("ticker_sentiment_score", 0)),
                    "sentiment_label": ticker_sentiment.get("ticker_sentiment_label", "Neutral")
                }
                
                news_items.append(news_item)
                
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Failed to parse news item: {e}")
                continue
        
        # Sort by time (most recent first)
        news_items.sort(key=lambda x: x["time_published"], reverse=True)
        
        # Cache results
        _cache[cache_key] = news_items
        _cache_timestamps[cache_key] = time.time()
        
        logger.info(f"Successfully fetched {len(news_items)} news items for {ticker}")
        return news_items
        
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        raise TimeoutError("Request to Alpha Vantage API timed out")
    
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
        raise ConnectionError(f"Failed to connect to Alpha Vantage API: {e}")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise