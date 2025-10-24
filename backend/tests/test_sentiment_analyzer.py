import pytest
from unittest.mock import patch, Mock


class TestSentimentAnalyzer:
    """Test suite for sentiment analysis functionality"""
    
    def test_analyze_sentiment_positive(self, sample_financial_texts):
        """Test sentiment analysis on positive financial text"""
        from app.sentiment_analyzer import analyze_sentiment
        
        positive_text = sample_financial_texts[0]["text"]
        
        with patch('app.sentiment_analyzer.get_sentiment_pipeline') as mock_pipeline:
            mock_pipeline.return_value = lambda x: [{"label": "positive", "score": 0.92}]
            
            result = analyze_sentiment(positive_text)
            
            assert result["label"] == "positive"
            assert result["score"] == 0.92
            assert "confidence" in result
            assert result["confidence"] == "high"  # score > 0.8
    
    def test_analyze_sentiment_negative(self, sample_financial_texts):
        """Test sentiment analysis on negative financial text"""
        from sentiment_analyzer import analyze_sentiment
        
        negative_text = sample_financial_texts[1]["text"]
        
        with patch('sentiment_analyzer.get_sentiment_pipeline') as mock_pipeline:
            mock_pipeline.return_value = lambda x: [{"label": "negative", "score": 0.88}]
            
            result = analyze_sentiment(negative_text)
            
            assert result["label"] == "negative"
            assert result["score"] == 0.88
            assert result["confidence"] == "high"
    
    def test_analyze_sentiment_neutral(self, sample_financial_texts):
        """Test sentiment analysis on neutral financial text"""
        from sentiment_analyzer import analyze_sentiment
        
        neutral_text = sample_financial_texts[2]["text"]
        
        with patch('sentiment_analyzer.get_sentiment_pipeline') as mock_pipeline:
            mock_pipeline.return_value = lambda x: [{"label": "neutral", "score": 0.75}]
            
            result = analyze_sentiment(neutral_text)
            
            assert result["label"] == "neutral"
            assert result["score"] == 0.75
            assert result["confidence"] == "medium"  # 0.6 < score <= 0.8
    
    def test_analyze_sentiment_low_confidence(self):
        """Test handling of low confidence predictions"""
        from sentiment_analyzer import analyze_sentiment
        
        with patch('sentiment_analyzer.get_sentiment_pipeline') as mock_pipeline:
            mock_pipeline.return_value = lambda x: [{"label": "neutral", "score": 0.45}]
            
            result = analyze_sentiment("Ambiguous market conditions persist.")
            
            assert result["confidence"] == "low"  # score <= 0.6
            assert "warning" in result
            assert "low confidence" in result["warning"].lower()
    
    def test_analyze_sentiment_empty_text(self):
        """Test handling of empty text input"""
        from sentiment_analyzer import analyze_sentiment
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            analyze_sentiment("")
    
    def test_analyze_sentiment_very_long_text(self):
        """Test handling of text exceeding model token limit"""
        from sentiment_analyzer import analyze_sentiment
        
        # Create text longer than 512 tokens (typical limit)
        long_text = "Apple stock " * 1000
        
        with patch('sentiment_analyzer.get_sentiment_pipeline') as mock_pipeline:
            mock_pipeline.return_value = lambda x: [{"label": "neutral", "score": 0.70}]
            
            result = analyze_sentiment(long_text)
            
            # Should truncate and still return result
            assert "label" in result
            assert "score" in result
    
    def test_batch_analyze_sentiment(self, sample_financial_texts):
        """Test batch sentiment analysis"""
        from sentiment_analyzer import batch_analyze_sentiment
        
        texts = [item["text"] for item in sample_financial_texts[:3]]
        
        with patch('sentiment_analyzer.get_sentiment_pipeline') as mock_pipeline:
            def mock_batch_predict(text_list):
                results = []
                for text in text_list:
                    if "surges" in text.lower() or "strong" in text.lower():
                        results.append({"label": "positive", "score": 0.90})
                    elif "lawsuit" in text.lower() or "decline" in text.lower():
                        results.append({"label": "negative", "score": 0.85})
                    else:
                        results.append({"label": "neutral", "score": 0.70})
                return results
            
            mock_pipeline.return_value = mock_batch_predict
            
            results = batch_analyze_sentiment(texts)
            
            assert len(results) == 3
            assert results[0]["label"] == "positive"
            assert results[1]["label"] == "negative"
            assert results[2]["label"] == "neutral"
    
    def test_model_initialization(self):
        """Test sentiment model initialization"""
        from sentiment_analyzer import initialize_model
        
        with patch('transformers.pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            
            model = initialize_model()
            
            # Verify correct model is loaded
            mock_pipeline.assert_called_once()
            call_args = mock_pipeline.call_args
            assert "sentiment-analysis" in str(call_args) or "text-classification" in str(call_args)
    
    def test_model_caching(self):
        """Test that model is cached after first load"""
        from sentiment_analyzer import get_sentiment_pipeline
        
        with patch('transformers.pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            
            # First call
            model1 = get_sentiment_pipeline()
            
            # Second call should return cached model
            model2 = get_sentiment_pipeline()
            
            # Pipeline should only be created once
            assert mock_pipeline.call_count == 1
            assert model1 is model2
    
    def test_sentiment_label_normalization(self):
        """Test normalization of different label formats"""
        from sentiment_analyzer import normalize_label
        
        # Test various label formats from different models
        assert normalize_label("POSITIVE") == "positive"
        assert normalize_label("positive") == "positive"
        assert normalize_label("LABEL_2") == "positive"  # Some models use LABEL_0, LABEL_1, LABEL_2
        assert normalize_label("bullish") == "positive"
        assert normalize_label("bearish") == "negative"
    
    def test_analyze_sentiment_with_context(self):
        """Test sentiment analysis considers financial context"""
        from sentiment_analyzer import analyze_sentiment
        
        # "Beat" in financial context is positive
        text = "Apple beats earnings estimates"
        
        with patch('sentiment_analyzer.get_sentiment_pipeline') as mock_pipeline:
            mock_pipeline.return_value = lambda x: [{"label": "positive", "score": 0.95}]
            
            result = analyze_sentiment(text)
            
            assert result["label"] == "positive"
            assert result["score"] > 0.8
    
    def test_handle_model_error(self):
        """Test graceful handling of model errors"""
        from sentiment_analyzer import analyze_sentiment
        
        with patch('sentiment_analyzer.get_sentiment_pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock(side_effect=RuntimeError("Model error"))
            
            with pytest.raises(RuntimeError, match="Model error"):
                analyze_sentiment("Test text")
    
    def test_confidence_thresholds(self):
        """Test confidence level categorization"""
        from sentiment_analyzer import get_confidence_level
        
        assert get_confidence_level(0.95) == "high"
        assert get_confidence_level(0.75) == "medium"
        assert get_confidence_level(0.55) == "low"
        assert get_confidence_level(0.40) == "low"