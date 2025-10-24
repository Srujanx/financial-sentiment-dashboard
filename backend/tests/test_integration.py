import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock


# Import app after adding to path in conftest.py
from main import app

client = TestClient(app)


class TestAPIIntegration:
    """Integration tests for the complete API workflow"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns basic info"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_health_check_endpoint(self):
        """Test health check endpoint"""
        with patch('sentiment_analyzer.get_sentiment_pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "model_loaded" in data
            assert "api_accessible" in data
            assert "version" in data
    
    def test_news_endpoint_success(self, mock_alpha_vantage_response):
        """Test successful news fetch and analysis"""
        with patch('requests.get') as mock_get, \
             patch('sentiment_analyzer.get_sentiment_pipeline') as mock_pipeline:
            
            # Mock Alpha Vantage response
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_alpha_vantage_response
            
            # Mock sentiment model
            mock_pipeline.return_value = lambda x: [{"label": "positive", "score": 0.9}]
            
            response = client.get("/news/AAPL")
            
            assert response.status_code == 200
            data = response.json()
            assert data["ticker"] == "AAPL"
            assert "total_items" in data
            assert "items" in data
            assert len(data["items"]) > 0
            
            # Verify sentiment analysis was performed
            first_item = data["items"][0]
            assert "ml_sentiment" in first_item
            assert "label" in first_item["ml_sentiment"]
            assert "score" in first_item["ml_sentiment"]
            assert "confidence" in first_item["ml_sentiment"]
    
    def test_news_endpoint_invalid_ticker(self):
        """Test news endpoint with invalid ticker"""
        response = client.get("/news/INVALID@")
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    def test_news_endpoint_api_error(self, mock_api_error_response):
        """Test handling of Alpha Vantage API errors"""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 400
            mock_get.return_value.json.return_value = mock_api_error_response
            
            response = client.get("/news/AAPL")
            
            assert response.status_code == 400
    
    def test_stats_endpoint(self, mock_alpha_vantage_response):
        """Test sentiment statistics endpoint"""
        with patch('requests.get') as mock_get, \
             patch('sentiment_analyzer.get_sentiment_pipeline') as mock_pipeline:
            
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_alpha_vantage_response
            
            # Mock different sentiments for each item
            sentiment_results = [
                [{"label": "positive", "score": 0.9}],
                [{"label": "negative", "score": 0.85}],
                [{"label": "neutral", "score": 0.7}]
            ]
            mock_pipeline.return_value = Mock(side_effect=sentiment_results)
            
            response = client.get("/news/AAPL/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert "total_news" in data
            assert "positive_count" in data
            assert "negative_count" in data
            assert "neutral_count" in data
            assert "sentiment_distribution" in data
            
            # Verify distribution adds up to 100%
            dist = data["sentiment_distribution"]
            total_pct = dist.get("positive", 0) + dist.get("negative", 0) + dist.get("neutral", 0)
            assert abs(total_pct - 100.0) < 0.1  # Allow for rounding
    
    def test_analyze_endpoint(self):
        """Test custom text analysis endpoint"""
        with patch('sentiment_analyzer.get_sentiment_pipeline') as mock_pipeline:
            mock_pipeline.return_value = lambda x: [{"label": "positive", "score": 0.95}]
            
            response = client.post(
                "/analyze",
                json={"text": "Apple stock surges on strong earnings"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["label"] == "positive"
            assert data["score"] > 0
            assert "confidence" in data
    
    def test_analyze_endpoint_empty_text(self):
        """Test analysis with empty text"""
        response = client.post("/analyze", json={"text": ""})
        
        assert response.status_code == 422  # Validation error
    
    def test_analyze_batch_endpoint(self):
        """Test batch analysis endpoint"""
        with patch('sentiment_analyzer.get_sentiment_pipeline') as mock_pipeline:
            def mock_batch(texts):
                return [{"label": "positive", "score": 0.9} for _ in texts]
            
            mock_pipeline.return_value = mock_batch
            
            response = client.post(
                "/analyze/batch",
                json={
                    "texts": [
                        "Stock rises on good news",
                        "Company announces strong earnings",
                        "Positive market sentiment continues"
                    ]
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 3
            assert data["total_analyzed"] == 3
    
    def test_cors_headers(self):
        """Test CORS headers are properly set"""
        response = client.options("/health")
        
        # FastAPI/Starlette handles OPTIONS automatically with CORS middleware
        assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly defined
    
    def test_end_to_end_workflow(self, mock_alpha_vantage_response):
        """Test complete workflow from news fetch to analysis"""
        with patch('requests.get') as mock_get, \
             patch('sentiment_analyzer.get_sentiment_pipeline') as mock_pipeline:
            
            # Setup mocks
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_alpha_vantage_response
            
            def mock_analyze(text):
                if "surges" in text.lower():
                    return [{"label": "positive", "score": 0.92}]
                elif "challenges" in text.lower():
                    return [{"label": "negative", "score": 0.88}]
                else:
                    return [{"label": "neutral", "score": 0.75}]
            
            mock_pipeline.return_value = mock_analyze
            
            # 1. Fetch news
            news_response = client.get("/news/AAPL")
            assert news_response.status_code == 200
            news_data = news_response.json()
            
            # 2. Verify sentiment analysis
            assert len(news_data["items"]) > 0
            for item in news_data["items"]:
                assert "ml_sentiment" in item
                sentiment = item["ml_sentiment"]
                assert sentiment["label"] in ["positive", "negative", "neutral"]
                assert 0 <= sentiment["score"] <= 1
            
            # 3. Get statistics
            stats_response = client.get("/news/AAPL/stats")
            assert stats_response.status_code == 200
            stats_data = stats_response.json()
            
            assert stats_data["total_news"] == len(news_data["items"])
            
            # 4. Analyze custom text
            custom_response = client.post(
                "/analyze",
                json={"text": news_data["items"][0]["title"]}
            )
            assert custom_response.status_code == 200
    
    def test_error_handling(self):
        """Test global error handling"""
        with patch('news_fetcher.fetch_financial_news') as mock_fetch:
            mock_fetch.side_effect = Exception("Unexpected error")
            
            response = client.get("/news/AAPL")
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
    
    def test_caching_behavior(self, mock_alpha_vantage_response):
        """Test that results are cached properly"""
        from news_fetcher import _clear_cache
        
        _clear_cache()
        
        with patch('requests.get') as mock_get, \
             patch('sentiment_analyzer.get_sentiment_pipeline') as mock_pipeline:
            
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_alpha_vantage_response
            mock_pipeline.return_value = lambda x: [{"label": "neutral", "score": 0.7}]
            
            # First request
            response1 = client.get("/news/AAPL")
            assert response1.status_code == 200
            
            # Second request should use cache
            response2 = client.get("/news/AAPL")
            assert response2.status_code == 200
            
            # Verify API was only called once due to caching
            assert mock_get.call_count == 1
    
    def test_concurrent_requests(self, mock_alpha_vantage_response):
        """Test handling of concurrent requests"""
        import concurrent.futures
        
        with patch('requests.get') as mock_get, \
             patch('sentiment_analyzer.get_sentiment_pipeline') as mock_pipeline:
            
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_alpha_vantage_response
            mock_pipeline.return_value = lambda x: [{"label": "neutral", "score": 0.7}]
            
            def make_request():
                return client.get("/news/AAPL")
            
            # Make 5 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request) for _ in range(5)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            # All requests should succeed
            assert all(r.status_code == 200 for r in results)