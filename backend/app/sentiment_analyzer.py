import logging
import threading
from typing import Dict, List
from transformers import pipeline

from app.config import get_settings
from app.models import ConfidenceLevel

logger = logging.getLogger(__name__)
settings = get_settings()

_model_pipeline = None
_model_lock = threading.Lock()  # Thread-safe loading


def normalize_label(label: str) -> str:
    label_lower = label.lower()
    label_mapping = {
        "positive": "positive",
        "bullish": "positive",
        "label_2": "positive",
        "negative": "negative",
        "bearish": "negative",
        "label_0": "negative",
        "neutral": "neutral",
        "label_1": "neutral"
    }
    return label_mapping.get(label_lower, "neutral")


def get_confidence_level(score: float) -> ConfidenceLevel:
    if score > 0.8:
        return ConfidenceLevel.HIGH
    elif score > 0.6:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


def initialize_model():
    logger.info(f"Loading sentiment model: {settings.SENTIMENT_MODEL}")
    try:
        model = pipeline("sentiment-analysis", model=settings.SENTIMENT_MODEL, device=-1)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model initialization failed: {e}")


def get_sentiment_pipeline():
    """
    Get sentiment pipeline with thread-safe lazy loading.
    Model loads only on first call to conserve memory at startup.
    """
    global _model_pipeline
    
    # Fast path: if model is already loaded, return it immediately
    if _model_pipeline is not None:
        return _model_pipeline
    
    # Slow path: need to load model with thread safety
    with _model_lock:
        # Double-check after acquiring lock (another thread might have loaded it)
        if _model_pipeline is None:
            logger.info("First sentiment analysis request - loading model now...")
            _model_pipeline = initialize_model()
            logger.info("Model ready for predictions")
        return _model_pipeline


def analyze_sentiment(text: str) -> Dict:
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    text = text.strip()
    
    if len(text) > settings.MAX_TEXT_LENGTH * 4:
        logger.warning("Text exceeds max length, truncating")
        text = text[:settings.MAX_TEXT_LENGTH * 4]
    
    try:
        # Model loads here on first call (lazy loading)
        pipeline_model = get_sentiment_pipeline()
        result = pipeline_model(text)[0]
        
        normalized_label = normalize_label(result["label"])
        score = result["score"]
        confidence = get_confidence_level(score)
        
        response = {
            "label": normalized_label,
            "score": round(score, 4),
            "confidence": confidence.value
        }
        
        if confidence == ConfidenceLevel.LOW:
            response["warning"] = "Low confidence prediction - results may be unreliable"
        
        logger.debug(f"Sentiment analysis: {normalized_label} (score: {score:.4f})")
        return response
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise RuntimeError(f"Sentiment analysis error: {e}")


def batch_analyze_sentiment(texts: List[str]) -> List[Dict]:
    if not texts:
        return []
    
    logger.info(f"Batch analyzing {len(texts)} texts")
    
    try:
        # Model loads here on first call (lazy loading)
        pipeline_model = get_sentiment_pipeline()
        
        processed_texts = []
        for text in texts:
            if len(text) > settings.MAX_TEXT_LENGTH * 4:
                text = text[:settings.MAX_TEXT_LENGTH * 4]
            processed_texts.append(text.strip())
        
        raw_results = pipeline_model(processed_texts)
        
        results = []
        for result in raw_results:
            normalized_label = normalize_label(result["label"])
            score = result["score"]
            confidence = get_confidence_level(score)
            
            response = {
                "label": normalized_label,
                "score": round(score, 4),
                "confidence": confidence.value
            }
            
            if confidence == ConfidenceLevel.LOW:
                response["warning"] = "Low confidence prediction - results may be unreliable"
            
            results.append(response)
        
        logger.info(f"Successfully analyzed {len(results)} texts")
        return results
        
    except Exception as e:
        logger.error(f"Batch sentiment analysis failed: {e}")
        raise RuntimeError(f"Batch sentiment analysis error: {e}")