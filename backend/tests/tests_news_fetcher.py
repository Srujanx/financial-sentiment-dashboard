import pytest
from unittest.mock import patch, Mock
import requests
from datetime import datetime


class TestNewsFetcher:
    """Test suite for news fetching functionality"""
    
    def test_fetch_financial_news_success(self, mock_alpha_vantage_response):
        """Test successful news fetch from Alpha Vantage"""
        from news_fetcher import fetch_financial_news
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_alpha_vantage_response
            
            result = fetch_financial_news(api_key="test_key", ticker="AAPL")
            
            # Verify API was called correctly
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "NEWS_SENTIMENT" in str(call_args)
            assert "AAPL" in str(call_args)
            
            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == 3
            
            # Verify first news item
            first_item = result[0]
            assert first_item["title"] == "Apple Reports Record Q4 Earnings, Stock Surges"
            assert first_item["ticker"] == "AAPL"
            assert "sentiment_score" in first_item
            assert "sentiment_label" in first_item
    
    def test_fetch_financial_news_api_error(self, mock_api_error_response):
        """Test handling of API errors"""
        from news_fetcher import fetch_financial_news
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 400
            mock_get.return_value.json.return_value = mock_api_error_response
            
            with pytest.raises(ValueError, match="API Error"):
                fetch_financial_news(api_key="invalid_key", ticker="AAPL")
    
    def test_fetch_financial_news_rate_limit(self, mock_rate_limit_response):
        """Test handling of rate limit errors"""
        from news_fetcher import fetch_financial_news
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_rate_limit_response
            
            with pytest.raises(ValueError, match="Rate limit"):
                fetch_financial_news(api_key="test_key", ticker="AAPL")
    
    def test_fetch_financial_news_network_error(self):
        """Test handling of network errors"""
        from news_fetcher import fetch_financial_news
        
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Network error")
            
            with pytest.raises(ConnectionError):
                fetch_financial_news(api_key="test_key", ticker="AAPL")
    
    def test_fetch_financial_news_timeout(self):
        """Test handling of timeout errors"""
        from news_fetcher import fetch_financial_news
        
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("Request timeout")
            
            with pytest.raises(TimeoutError):
                fetch_financial_news(api_key="test_key", ticker="AAPL")
    
    def test_fetch_financial_news_empty_response(self):
        """Test handling of empty news feed"""
        from news_fetcher import fetch_financial_news
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"feed": []}
            
            result = fetch_financial_news(api_key="test_key", ticker="AAPL")
            assert result == []
    
    def test_fetch_financial_news_filters_low_relevance(self, mock_alpha_vantage_response):
        """Test that low relevance news items are filtered out"""
        from news_fetcher import fetch_financial_news
        
        # Modify response to include low relevance item
        response = mock_alpha_vantage_response.copy()
        response["feed"].append({
            "title": "Unrelated Tech News",
            "summary": "General tech industry news",
            "ticker_sentiment": [
                {
                    "ticker": "AAPL",
                    "relevance_score": "0.1",  # Low relevance
                    "ticker_sentiment_score": "0.0",
                    "ticker_sentiment_label": "Neutral"
                }
            ]
        })
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = response
            
            result = fetch_financial_news(api_key="test_key", ticker="AAPL", min_relevance=0.4)
            
            # Should only return items with relevance >= 0.4
            assert len(result) == 3  # Original 3 items, not the low relevance one
            assert all(float(item["relevance_score"]) >= 0.4 for item in result)
    
    def test_fetch_financial_news_caching(self, mock_alpha_vantage_response):
        """Test that results are cached properly"""
        from news_fetcher import fetch_financial_news, _clear_cache
        
        # Clear any existing cache
        _clear_cache()
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_alpha_vantage_response
            
            # First call
            result1 = fetch_financial_news(api_key="test_key", ticker="AAPL")
            
            # Second call should use cache
            result2 = fetch_financial_news(api_key="test_key", ticker="AAPL")
            
            # API should only be called once
            assert mock_get.call_count == 1
            assert result1 == result2
    
    def test_parse_timestamp(self):
        """Test timestamp parsing from Alpha Vantage format"""
        from news_fetcher import parse_timestamp
        
        timestamp = "20241020T150000"
        result = parse_timestamp(timestamp)
        
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 10
        assert result.day == 20
        assert result.hour == 15
    
    def test_invalid_ticker(self):
        """Test handling of invalid ticker symbols"""
        from news_fetcher import fetch_financial_news
        
        with pytest.raises(ValueError, match="Invalid ticker"):
            fetch_financial_news(api_key="test_key", ticker="INVALID@SYMBOL")