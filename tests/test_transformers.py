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


# =============================================================================
# TEST: ATTENTION VISUALIZATION
# =============================================================================

class TestAttentionHead:
    """Tests for AttentionHead dataclass."""
    
    def test_attention_head_creation(self):
        """Test creating an attention head."""
        from jsf.ml.transformers.attention import AttentionHead
        
        weights = np.random.rand(5, 5)
        head = AttentionHead(layer=0, head=0, weights=weights)
        
        assert head.layer == 0
        assert head.head == 0
        assert head.shape == (5, 5)
    
    def test_get_attention_to_token(self):
        """Test getting attention TO a specific token."""
        from jsf.ml.transformers.attention import AttentionHead
        
        weights = np.eye(5)  # Identity matrix - each token attends to itself
        head = AttentionHead(layer=0, head=0, weights=weights)
        
        attention = head.get_attention_to_token(2)
        assert attention.shape == (5,)
        # In identity matrix, column 2 is all zeros except position 2
        assert attention[2] == 1.0
        assert attention[0] == 0.0
    
    def test_get_attention_from_token(self):
        """Test getting attention FROM a specific token."""
        from jsf.ml.transformers.attention import AttentionHead
        
        weights = np.eye(5)
        head = AttentionHead(layer=0, head=0, weights=weights)
        
        attention = head.get_attention_from_token(3)
        assert attention.shape == (5,)
        # In identity matrix, row 3 is all zeros except position 3
        assert attention[3] == 1.0
        assert attention[1] == 0.0


class TestAttentionPattern:
    """Tests for AttentionPattern container."""
    
    def test_pattern_creation(self):
        """Test creating an attention pattern."""
        from jsf.ml.transformers.attention import AttentionHead, AttentionPattern
        
        tokens = ["stock", "price", "rose", "sharply"]
        heads = [
            AttentionHead(layer=0, head=0, weights=np.random.rand(4, 4)),
            AttentionHead(layer=0, head=1, weights=np.random.rand(4, 4)),
            AttentionHead(layer=1, head=0, weights=np.random.rand(4, 4)),
            AttentionHead(layer=1, head=1, weights=np.random.rand(4, 4)),
        ]
        
        pattern = AttentionPattern(tokens=tokens, heads=heads)
        
        assert pattern.num_layers == 2
        assert pattern.num_heads_per_layer == 2
    
    def test_get_layer_attention(self):
        """Test retrieving attention heads from a specific layer."""
        from jsf.ml.transformers.attention import AttentionHead, AttentionPattern
        
        heads = [
            AttentionHead(layer=0, head=0, weights=np.random.rand(3, 3)),
            AttentionHead(layer=0, head=1, weights=np.random.rand(3, 3)),
            AttentionHead(layer=1, head=0, weights=np.random.rand(3, 3)),
        ]
        
        pattern = AttentionPattern(tokens=["a", "b", "c"], heads=heads)
        
        layer_0_heads = pattern.get_layer_attention(0)
        assert len(layer_0_heads) == 2
        
        layer_1_heads = pattern.get_layer_attention(1)
        assert len(layer_1_heads) == 1
    
    def test_average_attention(self):
        """Test computing average attention."""
        from jsf.ml.transformers.attention import AttentionHead, AttentionPattern
        
        # Create two heads with known patterns
        weights1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        weights2 = np.array([[0.0, 1.0], [1.0, 0.0]])
        
        heads = [
            AttentionHead(layer=0, head=0, weights=weights1),
            AttentionHead(layer=0, head=1, weights=weights2),
        ]
        
        pattern = AttentionPattern(tokens=["a", "b"], heads=heads)
        avg = pattern.average_attention()
        
        # Average should be uniform
        expected = np.array([[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_array_almost_equal(avg, expected)


class TestAttentionAnalyzer:
    """Tests for AttentionAnalyzer class."""
    
    def test_create_mock_pattern(self):
        """Test creating a mock attention pattern."""
        from jsf.ml.transformers.attention import AttentionAnalyzer
        
        analyzer = AttentionAnalyzer()
        pattern = analyzer.create_mock_pattern(
            tokens=["revenue", "increased", "by", "10%"],
            num_layers=2,
            num_heads=4,
            seed=42
        )
        
        assert len(pattern.tokens) == 4
        assert pattern.num_layers == 2
        assert pattern.num_heads_per_layer == 4
        assert len(pattern.heads) == 8  # 2 layers * 4 heads
        
        # Verify attention weights are normalized (rows sum to 1)
        for head in pattern.heads:
            row_sums = head.weights.sum(axis=1)
            np.testing.assert_array_almost_equal(row_sums, np.ones(4))
    
    def test_compute_attention_stats(self):
        """Test computing attention statistics."""
        from jsf.ml.transformers.attention import AttentionAnalyzer
        
        analyzer = AttentionAnalyzer()
        pattern = analyzer.create_mock_pattern(
            tokens=["Apple", "stock", "surged"],
            num_layers=2,
            num_heads=2,
            seed=123
        )
        
        stats = analyzer.compute_attention_stats(pattern)
        
        assert "num_layers" in stats
        assert "num_heads" in stats
        assert "seq_length" in stats
        assert "mean_entropy" in stats
        assert "token_importance" in stats
        assert "most_attended_token" in stats
        
        assert stats["num_layers"] == 2
        assert stats["seq_length"] == 3
        assert stats["most_attended_token"] in ["Apple", "stock", "surged"]
    
    def test_find_key_attention_pairs(self):
        """Test finding token pairs with high attention."""
        from jsf.ml.transformers.attention import AttentionHead, AttentionPattern, AttentionAnalyzer
        
        # Create pattern with known high attention
        weights = np.array([
            [0.1, 0.8, 0.1],  # Token 0 attends strongly to token 1
            [0.1, 0.1, 0.8],  # Token 1 attends strongly to token 2
            [0.8, 0.1, 0.1],  # Token 2 attends strongly to token 0
        ])
        
        heads = [AttentionHead(layer=0, head=0, weights=weights)]
        pattern = AttentionPattern(tokens=["A", "B", "C"], heads=heads)
        
        analyzer = AttentionAnalyzer()
        pairs = analyzer.find_key_attention_pairs(pattern, threshold=0.5)
        
        assert len(pairs) == 3
        # Check that high attention pairs are found
        pair_tokens = [(p[0], p[1]) for p in pairs]
        assert ("A", "B") in pair_tokens
        assert ("B", "C") in pair_tokens
        assert ("C", "A") in pair_tokens
    
    def test_attention_to_text_heatmap(self):
        """Test generating heatmap data."""
        from jsf.ml.transformers.attention import AttentionAnalyzer
        
        analyzer = AttentionAnalyzer()
        pattern = analyzer.create_mock_pattern(
            tokens=["market", "fell", "sharply"],
            num_layers=1,
            num_heads=2,
            seed=456
        )
        
        heatmap_data = analyzer.attention_to_text_heatmap(pattern)
        
        assert "tokens" in heatmap_data
        assert "attention_matrix" in heatmap_data
        assert "shape" in heatmap_data
        
        assert heatmap_data["tokens"] == ["market", "fell", "sharply"]
        assert len(heatmap_data["attention_matrix"]) == 3


class TestFinancialAttentionInterpreter:
    """Tests for FinancialAttentionInterpreter."""
    
    def test_categorize_tokens(self):
        """Test token categorization."""
        from jsf.ml.transformers.attention import FinancialAttentionInterpreter
        
        interpreter = FinancialAttentionInterpreter()
        tokens = ["Apple", "revenue", "increased", "15", "%", "in", "Q2"]
        
        categories = interpreter.categorize_tokens(tokens)
        
        # Check that categories exist
        assert "metrics" in categories
        assert "sentiment" in categories
        assert "numeric" in categories
        
        # Check categorization
        metric_tokens = [t for _, t in categories["metrics"]]
        assert "revenue" in metric_tokens
        
        sentiment_tokens = [t for _, t in categories["sentiment"]]
        assert "increased" in sentiment_tokens
        
        numeric_tokens = [t for _, t in categories["numeric"]]
        assert "15" in numeric_tokens
    
    def test_interpret_financial_attention(self):
        """Test financial attention interpretation."""
        from jsf.ml.transformers.attention import FinancialAttentionInterpreter
        
        interpreter = FinancialAttentionInterpreter()
        tokens = ["earnings", "beat", "expectations"]
        
        # Create attention that focuses on "beat"
        weights = np.array([
            [0.2, 0.6, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.6, 0.2],
        ])
        
        result = interpreter.interpret_financial_attention(tokens, weights)
        
        assert "token_categories" in result
        assert "attention_by_category" in result
        assert "top_attended_tokens" in result
        assert "interpretation" in result
        
        # "beat" should be most attended
        top_token = result["top_attended_tokens"][0][0]
        assert top_token == "beat"


class TestAnalyzeAttentionFunction:
    """Tests for the convenience analyze_attention function."""
    
    def test_analyze_attention_basic(self):
        """Test basic attention analysis."""
        from jsf.ml.transformers.attention import analyze_attention
        
        tokens = ["stock", "price", "increased"]
        weights = np.random.rand(3, 3)
        # Normalize rows to sum to 1
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        result = analyze_attention(tokens, weights, is_financial=True)
        
        assert "stats" in result
        assert "key_pairs" in result
        assert "financial" in result
    
    def test_analyze_attention_non_financial(self):
        """Test attention analysis without financial interpretation."""
        from jsf.ml.transformers.attention import analyze_attention
        
        tokens = ["hello", "world"]
        weights = np.array([[0.5, 0.5], [0.5, 0.5]])
        
        result = analyze_attention(tokens, weights, is_financial=False)
        
        assert "stats" in result
        assert "key_pairs" in result
        assert "financial" not in result
