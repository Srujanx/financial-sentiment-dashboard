from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class SentimentLabel(str, Enum):
    """Sentiment label enumeration"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ConfidenceLevel(str, Enum):
    """Confidence level enumeration"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SentimentResult(BaseModel):
    """Sentiment analysis result"""
    label: SentimentLabel
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    confidence: ConfidenceLevel
    warning: Optional[str] = None
    
    class Config:
        protected_namespaces = ()
    
    @validator('confidence', always=True)
    def determine_confidence(cls, v, values):
        """Determine confidence level based on score"""
        if 'score' in values:
            score = values['score']
            if score > 0.8:
                return ConfidenceLevel.HIGH
            elif score > 0.6:
                return ConfidenceLevel.MEDIUM
            else:
                return ConfidenceLevel.LOW
        return v


class NewsItem(BaseModel):
    """Individual news item"""
    title: str
    summary: str
    url: str
    source: str
    time_published: datetime
    ticker: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    sentiment_label: str
    
    class Config:
        protected_namespaces = ()
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class NewsItemWithAnalysis(NewsItem):
    """News item with additional ML analysis"""
    ml_sentiment: SentimentResult
    text_analyzed: str


class NewsFeedResponse(BaseModel):
    """Response model for news feed endpoint"""
    ticker: str
    total_items: int
    items: List[NewsItemWithAnalysis]
    timestamp: datetime = Field(default_factory=datetime.now)
    cache_hit: bool = False
    
    class Config:
        protected_namespaces = ()
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SentimentAnalysisRequest(BaseModel):
    """Request model for sentiment analysis"""
    text: str = Field(..., min_length=1, max_length=5000)
    
    class Config:
        protected_namespaces = ()
    
    @validator('text')
    def validate_text(cls, v):
        """Validate text is not empty or whitespace"""
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()


class BatchSentimentAnalysisRequest(BaseModel):
    """Request model for batch sentiment analysis"""
    texts: List[str] = Field(..., min_items=1, max_items=50)
    
    class Config:
        protected_namespaces = ()
    
    @validator('texts')
    def validate_texts(cls, v):
        """Validate all texts are non-empty"""
        cleaned = [text.strip() for text in v if text.strip()]
        if not cleaned:
            raise ValueError("At least one valid text is required")
        return cleaned


class BatchSentimentAnalysisResponse(BaseModel):
    """Response model for batch sentiment analysis"""
    results: List[SentimentResult]
    total_analyzed: int
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        protected_namespaces = ()


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    model_loaded: bool
    api_accessible: bool
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        protected_namespaces = ()


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        protected_namespaces = ()


class SentimentStats(BaseModel):
    """Sentiment statistics for dashboard"""
    total_news: int
    positive_count: int
    negative_count: int
    neutral_count: int
    avg_sentiment_score: float
    avg_relevance_score: float
    sentiment_distribution: dict
    
    class Config:
        protected_namespaces = ()
    
    @validator('sentiment_distribution', always=True)
    def calculate_distribution(cls, v, values):
        """Calculate percentage distribution"""
        total = values.get('total_news', 0)
        if total == 0:
            return {"positive": 0, "negative": 0, "neutral": 0}
        
        return {
            "positive": round((values.get('positive_count', 0) / total) * 100, 2),
            "negative": round((values.get('negative_count', 0) / total) * 100, 2),
            "neutral": round((values.get('neutral_count', 0) / total) * 100, 2)
        }