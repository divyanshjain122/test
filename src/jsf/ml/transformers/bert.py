"""
BERT Models for Financial Natural Language Processing.

This module provides transformer-based models specifically tuned for
financial text analysis, including:
- FinBERT: Pre-trained BERT for financial sentiment
- Custom BERT: Fine-tunable BERT for trading signals
- BERT Feature Extractor: Convert text to embeddings for ML

⚠️ EDUCATIONAL USE ONLY - Not for production trading.

FinBERT is a BERT model pre-trained on financial communication text
and fine-tuned for financial sentiment classification. It achieves
state-of-the-art performance on financial sentiment analysis tasks.

References:
    - FinBERT: https://github.com/ProsusAI/finBERT
    - BERT: https://arxiv.org/abs/1810.04805
    - Financial PhraseBank: Malo et al. (2014)

Example:
    >>> from jsf.ml.transformers import FinBERT
    >>> 
    >>> # Initialize FinBERT model
    >>> model = FinBERT(model_name='ProsusAI/finbert')
    >>> 
    >>> # Analyze sentiment
    >>> texts = [
    ...     "Company reports strong quarterly earnings",
    ...     "Stock plummets amid regulatory concerns"
    ... ]
    >>> results = model.predict(texts)
    >>> 
    >>> for text, result in zip(texts, results):
    ...     print(f"{text}: {result.label} ({result.score:.2f})")

Requirements:
    pip install transformers torch
    # OR
    pip install transformers tensorflow
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union, Tuple
from enum import Enum
import numpy as np


@dataclass
class BERTConfig:
    """Configuration for BERT models."""
    
    model_name: str = "ProsusAI/finbert"
    max_length: int = 512
    batch_size: int = 8
    device: str = "auto"  # 'auto', 'cpu', 'cuda', 'mps'
    use_fast_tokenizer: bool = True
    return_attention: bool = False
    cache_dir: Optional[str] = None


class SentimentLabel(Enum):
    """Sentiment classification labels."""
    
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class BERTSentimentResult:
    """Result from BERT sentiment analysis."""
    
    text: str
    label: SentimentLabel
    score: float
    probabilities: Dict[str, float]
    attention_weights: Optional[np.ndarray] = None


def _check_transformers():
    """Check if transformers is available."""
    try:
        import transformers
        return transformers
    except ImportError:
        raise ImportError(
            "HuggingFace Transformers is not installed.\n"
            "Install with: pip install transformers torch\n"
            "Or: pip install transformers tensorflow"
        )


def _get_device(device_str: str = "auto"):
    """Determine the best available device."""
    if device_str != "auto":
        return device_str
    
    # Try to import torch first
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        pass
    
    # Try TensorFlow
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            return "gpu"
        else:
            return "cpu"
    except ImportError:
        pass
    
    return "cpu"


class FinBERT:
    """
    FinBERT model for financial sentiment analysis.
    
    FinBERT is a BERT-based model pre-trained on financial text
    (earnings reports, financial news, analyst reports) and fine-tuned
    for financial sentiment classification.
    
    The model classifies text into three categories:
    - Positive: Bullish, favorable news
    - Negative: Bearish, unfavorable news
    - Neutral: Factual, no clear sentiment
    
    Example:
        >>> finbert = FinBERT()
        >>> 
        >>> # Single prediction
        >>> result = finbert.predict_one(
        ...     "Revenue increased 15% year-over-year"
        ... )
        >>> print(f"Sentiment: {result.label.value}")
        >>> print(f"Confidence: {result.score:.2%}")
        >>> 
        >>> # Batch prediction
        >>> texts = [
        ...     "Earnings beat expectations",
        ...     "Company faces bankruptcy risk",
        ...     "Quarterly results in line with forecast"
        ... ]
        >>> results = finbert.predict(texts)
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        config: Optional[BERTConfig] = None,
        use_mock: bool = False,
    ):
        """
        Initialize FinBERT model.
        
        Args:
            model_name: HuggingFace model identifier
            config: Optional BERTConfig for customization
            use_mock: Use mock predictions (for testing/demo)
        """
        self.config = config or BERTConfig(model_name=model_name)
        self.use_mock = use_mock
        
        if not use_mock:
            self._load_model()
        else:
            self.model = None
            self.tokenizer = None
            self.device = "cpu"
    
    def _load_model(self):
        """Load the FinBERT model and tokenizer."""
        transformers = _check_transformers()
        
        self.device = _get_device(self.config.device)
        
        # Load tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=self.config.use_fast_tokenizer,
            cache_dir=self.config.cache_dir,
        )
        
        # Load model
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            output_attentions=self.config.return_attention,
        )
        
        # Move to device
        if self.device == "cuda":
            self.model = self.model.cuda()
        elif self.device == "mps":
            self.model = self.model.to("mps")
        
        self.model.eval()
    
    def predict_one(
        self,
        text: str,
        return_attention: bool = False
    ) -> BERTSentimentResult:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text
            return_attention: Whether to return attention weights
        
        Returns:
            BERTSentimentResult with prediction
        """
        results = self.predict([text], return_attention=return_attention)
        return results[0]
    
    def predict(
        self,
        texts: List[str],
        return_attention: bool = False
    ) -> List[BERTSentimentResult]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of input texts
            return_attention: Whether to return attention weights
        
        Returns:
            List of BERTSentimentResult
        """
        if self.use_mock:
            return self._mock_predict(texts)
        
        import torch
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            elif self.device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get probabilities
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
            # Get attention if requested
            attentions = None
            if return_attention and outputs.attentions is not None:
                # Average attention across all heads and layers
                attentions = torch.stack(outputs.attentions).mean(dim=(0, 1)).cpu().numpy()
            
            # Create results
            for idx, text in enumerate(batch_texts):
                prob_dist = probs[idx]
                
                # Map to sentiment labels (FinBERT specific)
                # Label mapping: 0=positive, 1=negative, 2=neutral
                label_idx = int(np.argmax(prob_dist))
                labels = [SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL]
                label = labels[label_idx]
                
                probabilities = {
                    "positive": float(prob_dist[0]),
                    "negative": float(prob_dist[1]),
                    "neutral": float(prob_dist[2]),
                }
                
                attention_weights = attentions[idx] if attentions is not None else None
                
                results.append(BERTSentimentResult(
                    text=text,
                    label=label,
                    score=float(prob_dist[label_idx]),
                    probabilities=probabilities,
                    attention_weights=attention_weights,
                ))
        
        return results
    
    def _mock_predict(self, texts: List[str]) -> List[BERTSentimentResult]:
        """Generate mock predictions for testing."""
        # Use SimpleSentiment for mock
        from .sentiment import SimpleSentiment
        
        simple = SimpleSentiment()
        simple_results = simple.analyze(texts)
        
        # Convert to BERT format
        results = []
        for sr in simple_results:
            # Map simple sentiment to BERT labels
            if sr.score > 0.3:
                label = SentimentLabel.POSITIVE
                probs = {"positive": 0.7, "negative": 0.1, "neutral": 0.2}
            elif sr.score < -0.3:
                label = SentimentLabel.NEGATIVE
                probs = {"positive": 0.1, "negative": 0.7, "neutral": 0.2}
            else:
                label = SentimentLabel.NEUTRAL
                probs = {"positive": 0.2, "negative": 0.2, "neutral": 0.6}
            
            results.append(BERTSentimentResult(
                text=sr.text,
                label=label,
                score=probs[label.value],
                probabilities=probs,
            ))
        
        return results
    
    def get_embeddings(
        self,
        texts: List[str],
        layer: int = -1
    ) -> np.ndarray:
        """
        Extract BERT embeddings from texts.
        
        Args:
            texts: List of input texts
            layer: Which layer to extract from (-1 = last layer)
        
        Returns:
            Array of shape (n_texts, hidden_size)
        """
        if self.use_mock:
            # Return mock embeddings
            return np.random.randn(len(texts), 768).astype(np.float32)
        
        import torch
        
        embeddings = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            elif self.device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.bert(**inputs, output_hidden_states=True)
            
            # Get hidden states from specified layer
            hidden_states = outputs.hidden_states[layer]
            
            # Use [CLS] token embedding (first token)
            cls_embeddings = hidden_states[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
        
        return np.vstack(embeddings)


class CustomBERT:
    """
    Custom BERT model for fine-tuning on trading-specific tasks.
    
    This class allows fine-tuning a BERT model on custom financial
    datasets for tasks like:
    - News impact prediction
    - Price movement classification
    - Earnings call sentiment
    - SEC filing analysis
    
    Example:
        >>> # Create custom BERT for price prediction
        >>> bert = CustomBERT(
        ...     model_name='bert-base-uncased',
        ...     num_labels=3  # up, down, neutral
        ... )
        >>> 
        >>> # Fine-tune on custom data
        >>> bert.train(
        ...     texts=train_texts,
        ...     labels=train_labels,
        ...     epochs=3
        ... )
        >>> 
        >>> # Predict
        >>> predictions = bert.predict(test_texts)
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 3,
        config: Optional[BERTConfig] = None,
        use_mock: bool = False,
    ):
        """
        Initialize custom BERT model.
        
        Args:
            model_name: Base BERT model to use
            num_labels: Number of output classes
            config: Optional BERTConfig
            use_mock: Use mock mode (for testing)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.config = config or BERTConfig(model_name=model_name)
        self.use_mock = use_mock
        
        if not use_mock:
            self._load_model()
        else:
            self.model = None
            self.tokenizer = None
            self.device = "cpu"
    
    def _load_model(self):
        """Load the BERT model for fine-tuning."""
        transformers = _check_transformers()
        
        self.device = _get_device(self.config.device)
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=self.config.use_fast_tokenizer,
            cache_dir=self.config.cache_dir,
        )
        
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            cache_dir=self.config.cache_dir,
        )
        
        if self.device == "cuda":
            self.model = self.model.cuda()
        elif self.device == "mps":
            self.model = self.model.to("mps")
    
    def train(
        self,
        texts: List[str],
        labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the BERT model.
        
        Args:
            texts: Training texts
            labels: Training labels (integers)
            val_texts: Validation texts
            val_labels: Validation labels
            epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Warmup steps for scheduler
            weight_decay: Weight decay for AdamW
        
        Returns:
            Dictionary with training history
        """
        if self.use_mock:
            return {"train_loss": [0.5, 0.3, 0.2], "val_loss": [0.6, 0.4, 0.3]}
        
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        # Prepare training data
        train_encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        train_labels_tensor = torch.tensor(labels)
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            train_labels_tensor
        )
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Prepare optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training loop
        history = {"train_loss": [], "val_loss": []}
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in train_loader:
                input_ids, attention_mask, batch_labels = batch
                
                if self.device == "cuda":
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    batch_labels = batch_labels.cuda()
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            history["train_loss"].append(avg_loss)
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return history
    
    def predict(
        self,
        texts: List[str],
        return_probabilities: bool = True
    ) -> Union[List[int], List[Tuple[int, np.ndarray]]]:
        """
        Predict labels for texts.
        
        Args:
            texts: Input texts
            return_probabilities: Whether to return probabilities
        
        Returns:
            List of predicted labels, or (label, probabilities) tuples
        """
        if self.use_mock:
            predictions = [np.random.randint(0, self.num_labels) for _ in texts]
            if return_probabilities:
                probs = [np.random.dirichlet([1] * self.num_labels) for _ in texts]
                return list(zip(predictions, probs))
            return predictions
        
        import torch
        
        self.model.eval()
        predictions = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits.cpu().numpy()
            probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
            
            for idx in range(len(batch_texts)):
                label = int(np.argmax(logits[idx]))
                if return_probabilities:
                    predictions.append((label, probs[idx]))
                else:
                    predictions.append(label)
        
        return predictions
    
    def save(self, path: str) -> None:
        """Save the fine-tuned model."""
        if not self.use_mock:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
    
    def load(self, path: str) -> None:
        """Load a fine-tuned model."""
        if not self.use_mock:
            transformers = _check_transformers()
            self.model = transformers.AutoModelForSequenceClassification.from_pretrained(path)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(path)
            
            if self.device == "cuda":
                self.model = self.model.cuda()


class BERTFeatureExtractor:
    """
    Extract features from text using BERT embeddings.
    
    Converts financial text into dense vector representations
    that can be used as input features for ML models.
    
    Example:
        >>> extractor = BERTFeatureExtractor()
        >>> 
        >>> texts = [
        ...     "Company announces merger",
        ...     "Earnings decline for third quarter"
        ... ]
        >>> 
        >>> # Get embeddings
        >>> embeddings = extractor.extract(texts)
        >>> print(embeddings.shape)  # (2, 768)
        >>> 
        >>> # Use in ML pipeline
        >>> from sklearn.linear_model import LogisticRegression
        >>> clf = LogisticRegression()
        >>> clf.fit(embeddings, labels)
    """
    
    def __init__(
        self,
        model: Optional[Union[FinBERT, CustomBERT]] = None,
        pooling: str = "cls",  # 'cls', 'mean', 'max'
        use_mock: bool = False,
    ):
        """
        Initialize feature extractor.
        
        Args:
            model: BERT model to use (FinBERT or CustomBERT)
            pooling: Pooling strategy for embeddings
            use_mock: Use mock embeddings
        """
        self.model = model or FinBERT(use_mock=use_mock)
        self.pooling = pooling
        self.use_mock = use_mock
    
    def extract(self, texts: List[str]) -> np.ndarray:
        """
        Extract features from texts.
        
        Args:
            texts: Input texts
        
        Returns:
            Array of shape (n_texts, embedding_dim)
        """
        if hasattr(self.model, 'get_embeddings'):
            return self.model.get_embeddings(texts)
        
        # Fallback: use predictions as features
        if self.use_mock:
            return np.random.randn(len(texts), 768).astype(np.float32)
        
        # Use sentiment probabilities as simple features
        results = self.model.predict(texts)
        features = []
        for result in results:
            if hasattr(result, 'probabilities'):
                feat = [
                    result.probabilities.get("positive", 0.0),
                    result.probabilities.get("negative", 0.0),
                    result.probabilities.get("neutral", 0.0),
                ]
            else:
                feat = [0.33, 0.33, 0.34]  # Uniform fallback
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def fit_transform(
        self,
        texts: List[str],
        reduce_dim: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract and optionally reduce dimensionality.
        
        Args:
            texts: Input texts
            reduce_dim: Target dimensionality (uses PCA if provided)
        
        Returns:
            Transformed features
        """
        embeddings = self.extract(texts)
        
        if reduce_dim and reduce_dim < embeddings.shape[1]:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=reduce_dim)
            embeddings = pca.fit_transform(embeddings)
        
        return embeddings


# Convenience function
def create_finbert(
    use_mock: bool = False,
    device: str = "auto"
) -> FinBERT:
    """
    Create a FinBERT model with sensible defaults.
    
    Args:
        use_mock: Use mock predictions (for testing)
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
    
    Returns:
        Initialized FinBERT model
    """
    config = BERTConfig(device=device)
    return FinBERT(config=config, use_mock=use_mock)
