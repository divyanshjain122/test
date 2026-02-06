"""Financial Sentiment Analysis Models.

This module provides sentiment analysis capabilities for financial text,
including news headlines, earnings reports, and social media content.

⚠️ EDUCATIONAL PURPOSE ONLY ⚠️
Sentiment analysis should be used as ONE factor among many in research.
It is NOT a standalone trading signal and NOT financial advice.

Models:
- SimpleSentiment: Rule-based baseline using financial lexicons
- FinBERTSentiment: Fine-tuned BERT for financial sentiment (requires transformers)
- DistilBERTSentiment: Lightweight BERT variant (requires transformers)

Example:
    >>> from jsf.ml.transformers.sentiment import SimpleSentiment
    >>> 
    >>> analyzer = SimpleSentiment()
    >>> scores = analyzer.analyze([
    ...     "Tesla beats Q4 earnings expectations",
    ...     "Oil prices crash on demand concerns"
    ... ])
    >>> print(scores)
    # [{'text': '...', 'sentiment': 0.65, 'label': 'positive'}, ...]
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings

import numpy as np
import pandas as pd


class SentimentLabel(Enum):
    """Sentiment classification labels."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""
    text: str
    score: float  # -1 to 1
    label: SentimentLabel
    confidence: float  # 0 to 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'score': self.score,
            'label': self.label.value,
            'confidence': self.confidence,
        }


# Financial sentiment lexicons for rule-based analysis
# These are simplified versions for educational purposes
POSITIVE_FINANCIAL_WORDS = {
    # Earnings & Performance
    'beat', 'beats', 'exceeded', 'exceeds', 'surpass', 'surpassed',
    'outperform', 'outperformed', 'strong', 'stronger', 'strongest',
    'record', 'high', 'higher', 'highest', 'growth', 'grew', 'grow',
    'profit', 'profitable', 'profitability', 'gain', 'gains', 'gained',
    'increase', 'increased', 'increases', 'rise', 'rises', 'rose', 'rising',
    'surge', 'surged', 'surges', 'rally', 'rallied', 'rallies',
    'boom', 'booming', 'soar', 'soared', 'soaring', 'jump', 'jumped',
    
    # Outlook
    'optimistic', 'bullish', 'positive', 'upgrade', 'upgraded',
    'buy', 'outperform', 'overweight', 'recommend', 'recommended',
    'opportunity', 'opportunities', 'potential', 'promising', 'robust',
    
    # Fundamentals
    'dividend', 'dividends', 'buyback', 'acquisition', 'expand',
    'expansion', 'innovation', 'breakthrough', 'success', 'successful',
}

NEGATIVE_FINANCIAL_WORDS = {
    # Earnings & Performance
    'miss', 'missed', 'misses', 'below', 'weak', 'weaker', 'weakest',
    'underperform', 'underperformed', 'loss', 'losses', 'lost',
    'decline', 'declined', 'declines', 'declining', 'fall', 'fell', 'falls',
    'drop', 'dropped', 'drops', 'dropping', 'decrease', 'decreased',
    'plunge', 'plunged', 'plunges', 'crash', 'crashed', 'crashes',
    'collapse', 'collapsed', 'sink', 'sank', 'slump', 'slumped',
    
    # Outlook
    'pessimistic', 'bearish', 'negative', 'downgrade', 'downgraded',
    'sell', 'underweight', 'underperform', 'concern', 'concerns', 'worried',
    'risk', 'risks', 'risky', 'threat', 'threats', 'warning', 'warns',
    
    # Problems
    'bankruptcy', 'bankrupt', 'default', 'defaulted', 'layoff', 'layoffs',
    'lawsuit', 'investigation', 'fraud', 'scandal', 'crisis',
    'recession', 'downturn', 'uncertainty', 'volatile', 'volatility',
}

INTENSIFIERS = {
    'very': 1.5,
    'extremely': 2.0,
    'significantly': 1.5,
    'substantially': 1.5,
    'sharply': 1.5,
    'dramatically': 1.8,
    'slightly': 0.5,
    'marginally': 0.5,
    'somewhat': 0.7,
}

NEGATORS = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere'}


class SimpleSentiment:
    """Simple rule-based financial sentiment analyzer.
    
    Uses financial lexicons to score sentiment. This is a baseline
    approach for educational purposes. For production, use FinBERT.
    
    Example:
        >>> analyzer = SimpleSentiment()
        >>> results = analyzer.analyze([
        ...     "Company reports record profits",
        ...     "Stock plunges on weak earnings"
        ... ])
    """
    
    def __init__(
        self,
        positive_words: Optional[set] = None,
        negative_words: Optional[set] = None,
        custom_lexicon: Optional[Dict[str, float]] = None,
    ):
        """Initialize sentiment analyzer.
        
        Args:
            positive_words: Custom positive word set (overrides default)
            negative_words: Custom negative word set (overrides default)
            custom_lexicon: Dict mapping words to scores (-1 to 1)
        """
        self.positive_words = positive_words or POSITIVE_FINANCIAL_WORDS
        self.negative_words = negative_words or NEGATIVE_FINANCIAL_WORDS
        self.custom_lexicon = custom_lexicon or {}
    
    def _preprocess(self, text: str) -> List[str]:
        """Preprocess text into tokens."""
        import re
        # Simple tokenization
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens
    
    def _score_text(self, text: str) -> float:
        """Score a single text string."""
        tokens = self._preprocess(text)
        
        if not tokens:
            return 0.0
        
        score = 0.0
        multiplier = 1.0
        negation = False
        
        for i, token in enumerate(tokens):
            # Check for negation
            if token in NEGATORS:
                negation = True
                continue
            
            # Check for intensifiers
            if token in INTENSIFIERS:
                multiplier = INTENSIFIERS[token]
                continue
            
            # Score words
            word_score = 0.0
            
            if token in self.custom_lexicon:
                word_score = self.custom_lexicon[token]
            elif token in self.positive_words:
                word_score = 1.0
            elif token in self.negative_words:
                word_score = -1.0
            
            # Apply modifiers
            if word_score != 0:
                if negation:
                    word_score *= -0.5  # Negation weakens sentiment
                    negation = False
                
                word_score *= multiplier
                multiplier = 1.0  # Reset after use
                
                score += word_score
        
        # Normalize by number of sentiment words
        sentiment_word_count = sum(
            1 for t in tokens 
            if t in self.positive_words or t in self.negative_words or t in self.custom_lexicon
        )
        
        if sentiment_word_count > 0:
            score = score / sentiment_word_count
        
        # Clamp to [-1, 1]
        score = max(-1.0, min(1.0, score))
        
        return score
    
    def _get_label(self, score: float) -> SentimentLabel:
        """Convert score to label."""
        if score > 0.2:
            return SentimentLabel.POSITIVE
        elif score < -0.2:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL
    
    def analyze(
        self, 
        texts: Union[str, List[str]]
    ) -> Union[SentimentResult, List[SentimentResult]]:
        """Analyze sentiment of text(s).
        
        Args:
            texts: Single text string or list of strings
            
        Returns:
            SentimentResult or list of SentimentResults
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        
        results = []
        for text in texts:
            score = self._score_text(text)
            label = self._get_label(score)
            
            # Confidence based on score magnitude
            confidence = min(abs(score) * 1.5, 1.0)
            
            result = SentimentResult(
                text=text,
                score=score,
                label=label,
                confidence=confidence,
            )
            results.append(result)
        
        return results[0] if single else results
    
    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        output_columns: bool = True,
    ) -> pd.DataFrame:
        """Analyze sentiment for a DataFrame column.
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text
            output_columns: If True, add score/label columns
            
        Returns:
            DataFrame with sentiment columns added
        """
        texts = df[text_column].tolist()
        results = self.analyze(texts)
        
        if output_columns:
            df = df.copy()
            df['sentiment_score'] = [r.score for r in results]
            df['sentiment_label'] = [r.label.value for r in results]
            df['sentiment_confidence'] = [r.confidence for r in results]
        
        return df


class FinBERTSentiment:
    """FinBERT-based financial sentiment analyzer.
    
    Uses the ProsusAI/finbert model fine-tuned on financial text.
    Requires: pip install transformers torch (or tensorflow)
    
    Example:
        >>> analyzer = FinBERTSentiment()
        >>> results = analyzer.analyze([
        ...     "The company exceeded quarterly expectations",
        ...     "Investors worried about rising interest rates"
        ... ])
        
    Note:
        First run will download the FinBERT model (~440MB).
        For faster inference, consider using DistilBERTSentiment.
    """
    
    MODEL_NAME = "ProsusAI/finbert"
    
    def __init__(
        self,
        model_name: str = None,
        device: str = "auto",
        max_length: int = 512,
    ):
        """Initialize FinBERT sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name (default: ProsusAI/finbert)
            device: Device to use ('cpu', 'cuda', 'auto')
            max_length: Maximum token length
        """
        self.model_name = model_name or self.MODEL_NAME
        self.device = device
        self.max_length = max_length
        
        self._model = None
        self._tokenizer = None
        self._pipeline = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._pipeline is not None:
            return
        
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            raise ImportError(
                "HuggingFace Transformers is not installed.\n"
                "Install with: pip install transformers torch\n"
                "Or: pip install transformers tensorflow"
            )
        
        # Determine device
        device = self.device
        if device == "auto":
            try:
                import torch
                device = 0 if torch.cuda.is_available() else -1
            except ImportError:
                device = -1  # CPU for TensorFlow
        
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            tokenizer=self.model_name,
            device=device,
            max_length=self.max_length,
            truncation=True,
        )
    
    def analyze(
        self, 
        texts: Union[str, List[str]]
    ) -> Union[SentimentResult, List[SentimentResult]]:
        """Analyze sentiment using FinBERT.
        
        Args:
            texts: Single text string or list of strings
            
        Returns:
            SentimentResult or list of SentimentResults
        """
        self._load_model()
        
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        
        # Get predictions
        predictions = self._pipeline(texts)
        
        results = []
        for text, pred in zip(texts, predictions):
            # Map FinBERT labels to our format
            label_map = {
                'positive': SentimentLabel.POSITIVE,
                'negative': SentimentLabel.NEGATIVE,
                'neutral': SentimentLabel.NEUTRAL,
            }
            
            label = label_map.get(pred['label'].lower(), SentimentLabel.NEUTRAL)
            confidence = pred['score']
            
            # Convert to score
            if label == SentimentLabel.POSITIVE:
                score = confidence
            elif label == SentimentLabel.NEGATIVE:
                score = -confidence
            else:
                score = 0.0
            
            result = SentimentResult(
                text=text,
                score=score,
                label=label,
                confidence=confidence,
            )
            results.append(result)
        
        return results[0] if single else results


# Convenience function
def analyze_sentiment(
    texts: Union[str, List[str]],
    model: str = "simple",
) -> List[SentimentResult]:
    """Analyze sentiment of financial texts.
    
    Args:
        texts: Text or list of texts to analyze
        model: 'simple' (rule-based) or 'finbert' (BERT-based)
        
    Returns:
        List of SentimentResult objects
        
    Example:
        >>> results = analyze_sentiment(
        ...     ["Stock rallies on strong earnings"],
        ...     model="simple"
        ... )
    """
    if model == "simple":
        analyzer = SimpleSentiment()
    elif model == "finbert":
        analyzer = FinBERTSentiment()
    else:
        raise ValueError(f"Unknown model: {model}. Use 'simple' or 'finbert'.")
    
    if isinstance(texts, str):
        texts = [texts]
    
    return analyzer.analyze(texts)


__all__ = [
    "SentimentLabel",
    "SentimentResult",
    "SimpleSentiment",
    "FinBERTSentiment",
    "analyze_sentiment",
    "POSITIVE_FINANCIAL_WORDS",
    "NEGATIVE_FINANCIAL_WORDS",
]
