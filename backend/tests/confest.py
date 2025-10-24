from unittest.mock import Mock , patch
import sys
from pathlib import Path
import pytest
# backend --> path 
backend_path = Path(__file__).parent.parent / "app"
sys.path.insert(0 , str(backend_path))

@pytest.fixture
def mock_alpha_vantage_response():
    return {
        
        "items": "50",
        "sentiment_score_definition": "x <= -0.35: Bearish; -0.35 < x <= -0.15: Somewhat-Bearish; -0.15 < x < 0.15: Neutral; 0.15 <= x < 0.35: Somewhat_Bullish; x >= 0.35: Bullish",
        "relevance_score_definition": "0 < x <= 1, with a higher score indicating higher relevance.",
        "feed": [
            {
                "title": "Apple Reports Record Q4 Earnings, Stock Surges",
                "url": "https://example.com/news1",
                "time_published": "20241020T150000",
                "authors": ["John Doe"],
                "summary": "Apple Inc. reported better-than-expected quarterly earnings...",
                "source": "Reuters",
                "overall_sentiment_score": 0.45,
                "overall_sentiment_label": "Bullish",
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "relevance_score": "0.95",
                        "ticker_sentiment_score": "0.48",
                        "ticker_sentiment_label": "Bullish"
                    }
                ]
            },
            {
                "title": "Apple Faces Regulatory Challenges in Europe",
                "url": "https://example.com/news2",
                "time_published": "20241020T120000",
                "authors": ["Jane Smith"],
                "summary": "European regulators announce new investigation into Apple's App Store practices...",
                "source": "Bloomberg",
                "overall_sentiment_score": -0.25,
                "overall_sentiment_label": "Somewhat-Bearish",
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "relevance_score": "0.88",
                        "ticker_sentiment_score": "-0.22",
                        "ticker_sentiment_label": "Somewhat-Bearish"
                    }
                ]
            },
            {
                "title": "Tech Industry Updates: Mixed Day for Major Stocks",
                "url": "https://example.com/news3",
                "time_published": "20241020T100000",
                "authors": ["Tech News Team"],
                "summary": "Tech stocks showed mixed performance today with Apple holding steady...",
                "source": "CNBC",
                "overall_sentiment_score": 0.05,
                "overall_sentiment_label": "Neutral",
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "relevance_score": "0.45",
                        "ticker_sentiment_score": "0.02",
                        "ticker_sentiment_label": "Neutral"
                    }
                ]
            }
        ]
    }

@pytest.fixture
# error response 
def mock_api_error_response():
    return {
        "Error Message": "Invalid API call."
    }
# rate limit
def mock_rate_limit_response():
    return {
        "Note" : "Our standard API call frequency is 25 requests per day"
    }
@pytest.fixture
def sample_financial_texts():
    return [
        { 
            "text": "Apple stock surges on record iPhone sales and strong guidance for next quarter.",
            "expected_label": "positive"
        },
        {
            "text": "Apple faces lawsuit over alleged antitrust violations, shares decline.",
            "expected_label": "negative"
        },
        {
            "text": "Apple announces quarterly dividend, maintains current stock buyback program.",
            "expected_label": "neutral"
        },
        {
            "text": "Analysts upgrade Apple rating following impressive earnings beat and innovation pipeline.",
            "expected_label": "positive"
        },
        {
            "text": "Apple warns of supply chain disruptions impacting production targets.",
            "expected_label": "negative"
        }
    ]
@pytest.fixture
def mock_sentiment_model():
    mock_model = Mock()
    
    def mock_predict(text):
        if any(word in text.lower() for word in ["surges", "record", "strong", "upgrade", "beat"]):
            return [{"label": "positive", "score": 0.92}]
        elif any(word in text.lower() for word in ["decline", "lawsuit", "warns", "disruptions"]):
            return [{"label": "negative", "score": 0.88}]
        else:
            return [{"label": "neutral", "score": 0.75}]
    
    mock_model.side_effect = mock_predict
    return mock_model

@pytest.fixture
def sample_news_items():
    return [ 
        {
            "title": "Apple Reports Record Earnings",
            "summary": "Strong quarterly performance exceeds expectations.",
            "url": "https://example.com/1",
            "source": "Reuters",
            "time_published": "20241020T150000",
            "sentiment_score": 0.45,
            "sentiment_label": "Bullish"
        },
        {
            "title": "Regulatory Challenges Mount",
            "summary": "New investigation announced by EU regulators.",
            "url": "https://example.com/2",
            "source": "Bloomberg",
            "time_published": "20241020T120000",
            "sentiment_score": -0.25,
            "sentiment_label": "Somewhat-Bearish"
        
        }
    ]
