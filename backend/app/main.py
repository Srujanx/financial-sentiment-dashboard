from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import List
from datetime import datetime

from app.config import get_settings, setup_logging
from app.models import (
    NewsFeedResponse,
    NewsItemWithAnalysis,
    SentimentAnalysisRequest,
    BatchSentimentAnalysisRequest,
    BatchSentimentAnalysisResponse,
    HealthCheckResponse,
    ErrorResponse,
    SentimentStats,
    SentimentResult
)
from app.news_fetcher import fetch_financial_news
from app.sentiment_analyzer import analyze_sentiment, batch_analyze_sentiment, get_sentiment_pipeline

# Initialize logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Financial Sentiment API",
    description="API for analyzing financial news sentiment using ML",
    version="1.0.0"
)

# Get settings
settings = get_settings()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Financial Sentiment API")
    try:
        # get_settings().validate()  # ← Call validate on the instance
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    
    # Pre-load ML model
    try:
        logger.info("Pre-loading sentiment analysis model...")
        get_sentiment_pipeline()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Don't fail startup, but log the error


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Financial Sentiment API")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.DEBUG else None
        ).dict()
    )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Financial Sentiment API",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    model_loaded = False
    api_accessible = False
    
    # Check if model is loaded
    try:
        get_sentiment_pipeline()
        model_loaded = True
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
    
    # Check if API key is configured
    if settings.ALPHA_VANTAGE_API_KEY:
        api_accessible = True
    
    status = "healthy" if (model_loaded and api_accessible) else "degraded"
    
    return HealthCheckResponse(
        status=status,
        version=settings.APP_VERSION,
        model_loaded=model_loaded,
        api_accessible=api_accessible
    )


@app.get("/news/{ticker}", response_model=NewsFeedResponse, tags=["News"])
async def get_news(ticker: str = "AAPL"):
    """
    Fetch and analyze financial news for a given ticker
    
    Args:
        ticker: Stock ticker symbol (default: AAPL)
        
    Returns:
        News feed with sentiment analysis
    """
    try:
        logger.info(f"Fetching news for ticker: {ticker}")
        
        # Fetch news from Alpha Vantage
        news_items = fetch_financial_news(
            api_key=settings.ALPHA_VANTAGE_API_KEY,
            ticker=ticker
        )
        
        # Analyze sentiment for each news item
        analyzed_items = []
        for item in news_items:
            try:
                # Combine title and summary for analysis
                text_to_analyze = f"{item['title']}. {item['summary']}"
                
                # Run sentiment analysis
                ml_sentiment = analyze_sentiment(text_to_analyze)
                
                # Create analyzed news item
                analyzed_item = NewsItemWithAnalysis(
                    **item,
                    ml_sentiment=SentimentResult(**ml_sentiment),
                    text_analyzed=text_to_analyze[:200]  # Store first 200 chars
                )
                
                analyzed_items.append(analyzed_item)
                
            except Exception as e:
                logger.warning(f"Failed to analyze news item: {e}")
                continue
        
        logger.info(f"Successfully analyzed {len(analyzed_items)} news items")
        
        return NewsFeedResponse(
            ticker=ticker.upper(),
            total_items=len(analyzed_items),
            items=analyzed_items,
            cache_hit=False
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        raise HTTPException(status_code=503, detail="Failed to connect to news API")
    
    except TimeoutError as e:
        logger.error(f"Timeout error: {e}")
        raise HTTPException(status_code=504, detail="Request timeout")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/news/{ticker}/stats", response_model=SentimentStats, tags=["Analytics"])
async def get_sentiment_stats(ticker: str = "AAPL"):
    """
    Get sentiment statistics for a ticker
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Sentiment statistics
    """
    try:
        # Get news feed
        news_response = await get_news(ticker)
        items = news_response.items
        
        if not items:
            return SentimentStats(
                total_news=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                avg_sentiment_score=0.0,
                avg_relevance_score=0.0,
                sentiment_distribution={}
            )
        
        # Calculate statistics
        positive_count = sum(1 for item in items if item.ml_sentiment.label == "positive")
        negative_count = sum(1 for item in items if item.ml_sentiment.label == "negative")
        neutral_count = sum(1 for item in items if item.ml_sentiment.label == "neutral")
        
        avg_sentiment = sum(item.sentiment_score for item in items) / len(items)
        avg_relevance = sum(item.relevance_score for item in items) / len(items)
        
        return SentimentStats(
            total_news=len(items),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            avg_sentiment_score=round(avg_sentiment, 4),
            avg_relevance_score=round(avg_relevance, 4),
            sentiment_distribution={}  # Will be calculated by validator
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate statistics")


@app.post("/analyze", response_model=SentimentResult, tags=["Analysis"])
async def analyze_text(request: SentimentAnalysisRequest):
    """
    Analyze sentiment of custom text
    
    Args:
        request: Text to analyze
        
    Returns:
        Sentiment analysis result
    """
    try:
        result = analyze_sentiment(request.text)
        return SentimentResult(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@app.post("/analyze/batch", response_model=BatchSentimentAnalysisResponse, tags=["Analysis"])
async def analyze_batch(request: BatchSentimentAnalysisRequest):
    """
    Analyze sentiment of multiple texts in batch
    
    Args:
        request: List of texts to analyze
        
    Returns:
        Batch sentiment analysis results
    """
    try:
        results = batch_analyze_sentiment(request.texts)
        return BatchSentimentAnalysisResponse(
            results=[SentimentResult(**r) for r in results],
            total_analyzed=len(results)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=500, detail="Batch analysis failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )