"""Tests for Transformer-based NLP Models.

Tests for sentiment analysis and text embedding functionality.
"""

import pytest
import pandas as pd
import numpy as np


# =============================================================================
# TEST: SIMPLE SENTIMENT ANALYZER
# =============================================================================

class TestSimpleSentiment:
    """Tests for rule-based sentiment analyzer."""
    
    def test_import(self):
        """Test that SimpleSentiment can be imported."""
        from jsf.ml.transformers.sentiment import SimpleSentiment
        analyzer = SimpleSentiment()
        assert analyzer is not None
    
    def test_positive_sentiment(self):
        """Test positive sentiment detection."""
        from jsf.ml.transformers.sentiment import SimpleSentiment, SentimentLabel
        
        analyzer = SimpleSentiment()
        
        # Clearly positive texts
        positive_texts = [
            "Company beats earnings expectations with record profit",
            "Stock surges on strong quarterly growth",
            "Investors bullish on expansion opportunities",
        ]
        
        results = analyzer.analyze(positive_texts)
        
        for result in results:
            assert result.score > 0, f"Expected positive score for: {result.text}"
            assert result.label == SentimentLabel.POSITIVE
    
    def test_negative_sentiment(self):
        """Test negative sentiment detection."""
        from jsf.ml.transformers.sentiment import SimpleSentiment, SentimentLabel
        
        analyzer = SimpleSentiment()
        
        # Clearly negative texts
        negative_texts = [
            "Stock crashes on weak earnings report",
            "Company reports unexpected loss and layoffs",
            "Market plunges amid recession fears",
        ]
        
        results = analyzer.analyze(negative_texts)
        
        for result in results:
            assert result.score < 0, f"Expected negative score for: {result.text}"
            assert result.label == SentimentLabel.NEGATIVE
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment detection."""
        from jsf.ml.transformers.sentiment import SimpleSentiment, SentimentLabel
        
        analyzer = SimpleSentiment()
        
        # Neutral texts
        neutral_texts = [
            "Company announces quarterly results",
            "CEO speaks at industry conference",
            "Market closes for holiday",
        ]
        
        results = analyzer.analyze(neutral_texts)
        
        for result in results:
            assert -0.3 <= result.score <= 0.3, f"Expected neutral score for: {result.text}"
    
    def test_single_text(self):
        """Test single text input."""
        from jsf.ml.transformers.sentiment import SimpleSentiment
        
        analyzer = SimpleSentiment()
        result = analyzer.analyze("Stock gains on positive news")
        
        # Should return single result, not list
        assert hasattr(result, 'score')
        assert hasattr(result, 'label')
    
    def test_negation_handling(self):
        """Test that negation flips sentiment."""
        from jsf.ml.transformers.sentiment import SimpleSentiment
        
        analyzer = SimpleSentiment()
        
        positive = analyzer.analyze("Stock gains").score
        negated = analyzer.analyze("Stock not gains").score
        
        # Negation should reduce or flip the sentiment
        assert negated < positive
    
    def test_intensifiers(self):
        """Test that intensifiers increase magnitude."""
        from jsf.ml.transformers.sentiment import SimpleSentiment
        
        analyzer = SimpleSentiment()
        
        normal = abs(analyzer.analyze("Stock gains").score)
        intensified = abs(analyzer.analyze("Stock dramatically gains").score)
        
        # Intensifier should increase magnitude
        assert intensified >= normal
    
    def test_dataframe_analysis(self):
        """Test DataFrame integration."""
        from jsf.ml.transformers.sentiment import SimpleSentiment
        
        analyzer = SimpleSentiment()
        
        df = pd.DataFrame({
            'headline': [
                "Company beats expectations",
                "Stock crashes on news",
                "Quarterly results announced",
            ],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        })
        
        result_df = analyzer.analyze_dataframe(df, text_column='headline')
        
        assert 'sentiment_score' in result_df.columns
        assert 'sentiment_label' in result_df.columns
        assert 'sentiment_confidence' in result_df.columns
        assert len(result_df) == 3
    
    def test_custom_lexicon(self):
        """Test custom lexicon support."""
        from jsf.ml.transformers.sentiment import SimpleSentiment
        
        custom = {
            'moon': 1.0,  # Crypto slang for bullish
            'rekt': -1.0,  # Crypto slang for loss
        }
        
        analyzer = SimpleSentiment(custom_lexicon=custom)
        
        result = analyzer.analyze("Stock going to moon")
        assert result.score > 0
    
    def test_result_to_dict(self):
        """Test SentimentResult.to_dict()."""
        from jsf.ml.transformers.sentiment import SimpleSentiment
        
        analyzer = SimpleSentiment()
        result = analyzer.analyze("Stock rises")
        
        d = result.to_dict()
        assert 'text' in d
        assert 'score' in d
        assert 'label' in d
        assert 'confidence' in d


# =============================================================================
# TEST: CONVENIENCE FUNCTION
# =============================================================================

class TestAnalyzeSentiment:
    """Tests for the analyze_sentiment convenience function."""
    
    def test_simple_model(self):
        """Test with simple model."""
        from jsf.ml.transformers.sentiment import analyze_sentiment
        
        results = analyze_sentiment(
            ["Stock gains on news", "Market drops"],
            model="simple"
        )
        
        assert len(results) == 2
        assert results[0].score > results[1].score
    
    def test_invalid_model(self):
        """Test error on invalid model."""
        from jsf.ml.transformers.sentiment import analyze_sentiment
        
        with pytest.raises(ValueError):
            analyze_sentiment("test", model="invalid_model")


# =============================================================================
# TEST: TEXT EMBEDDER
# =============================================================================

class TestTextEmbedder:
    """Tests for text embedding."""
    
    def test_import(self):
        """Test that TextEmbedder can be imported."""
        from jsf.ml.transformers import TextEmbedder
        embedder = TextEmbedder()
        assert embedder is not None
    
    def test_embed_shape(self):
        """Test embedding output shape."""
        from jsf.ml.transformers import TextEmbedder
        
        embedder = TextEmbedder(embedding_dim=768)
        texts = ["Hello world", "Test sentence"]
        
        embeddings = embedder.embed(texts)
        
        assert embeddings.shape == (2, 768)
    
    def test_embed_normalized(self):
        """Test that embeddings are normalized."""
        from jsf.ml.transformers import TextEmbedder
        
        embedder = TextEmbedder()
        texts = ["Test sentence"]
        
        embeddings = embedder.embed(texts)
        
        # Check unit norm
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-6
    
    def test_embed_deterministic(self):
        """Test that same text gives same embedding."""
        from jsf.ml.transformers import TextEmbedder
        
        embedder = TextEmbedder()
        
        emb1 = embedder.embed(["Same text"])
        emb2 = embedder.embed(["Same text"])
        
        np.testing.assert_array_almost_equal(emb1, emb2)


# =============================================================================
# TEST: SENTIMENT ANALYZER BASE CLASS
# =============================================================================

class TestSentimentAnalyzerBase:
    """Tests for base SentimentAnalyzer class."""
    
    def test_import(self):
        """Test that SentimentAnalyzer can be imported."""
        from jsf.ml.transformers import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        assert analyzer is not None
    
    def test_predict_mock(self):
        """Test mock prediction."""
        from jsf.ml.transformers import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer(model_name="mock")
        
        texts = ["Stock gains profit growth", "Stock crash loss decline"]
        scores = analyzer.predict(texts)
        
        assert len(scores) == 2
        assert scores[0] > scores[1]  # First should be more positive
    
    def test_predict_proba(self):
        """Test probability prediction."""
        from jsf.ml.transformers import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        
        probas = analyzer.predict_proba(["Good news"])
        
        assert len(probas) == 1
        assert 'positive' in probas[0]
        assert 'neutral' in probas[0]
        assert 'negative' in probas[0]


# =============================================================================
# TEST: LEXICON CONSTANTS
# =============================================================================

class TestLexicons:
    """Tests for financial lexicon constants."""
    
    def test_positive_words_exist(self):
        """Test positive words are defined."""
        from jsf.ml.transformers.sentiment import POSITIVE_FINANCIAL_WORDS
        
        assert len(POSITIVE_FINANCIAL_WORDS) > 50
        assert 'profit' in POSITIVE_FINANCIAL_WORDS
        assert 'growth' in POSITIVE_FINANCIAL_WORDS
        assert 'beat' in POSITIVE_FINANCIAL_WORDS
    
    def test_negative_words_exist(self):
        """Test negative words are defined."""
        from jsf.ml.transformers.sentiment import NEGATIVE_FINANCIAL_WORDS
        
        assert len(NEGATIVE_FINANCIAL_WORDS) > 50
        assert 'loss' in NEGATIVE_FINANCIAL_WORDS
        assert 'crash' in NEGATIVE_FINANCIAL_WORDS
        assert 'decline' in NEGATIVE_FINANCIAL_WORDS
    
    def test_no_overlap(self):
        """Test that positive and negative words don't overlap."""
        from jsf.ml.transformers.sentiment import (
            POSITIVE_FINANCIAL_WORDS,
            NEGATIVE_FINANCIAL_WORDS
        )
        
        overlap = POSITIVE_FINANCIAL_WORDS & NEGATIVE_FINANCIAL_WORDS
        assert len(overlap) == 0, f"Overlapping words: {overlap}"
