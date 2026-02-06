"""
Attention Visualization Utilities for Transformer Models.

This module provides tools for visualizing and analyzing attention patterns
in transformer-based models, helping understand what the model focuses on.

⚠️ EDUCATIONAL USE ONLY - Not for production trading.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class AttentionHead:
    """Represents attention weights from a single attention head."""
    
    layer: int
    head: int
    weights: np.ndarray  # Shape: (seq_len, seq_len)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of attention weights."""
        return self.weights.shape
    
    def get_attention_to_token(self, token_idx: int) -> np.ndarray:
        """Get attention weights TO a specific token position."""
        return self.weights[:, token_idx]
    
    def get_attention_from_token(self, token_idx: int) -> np.ndarray:
        """Get attention weights FROM a specific token position."""
        return self.weights[token_idx, :]


@dataclass
class AttentionPattern:
    """Collection of attention patterns from a transformer forward pass."""
    
    tokens: List[str]
    heads: List[AttentionHead]
    
    @property
    def num_layers(self) -> int:
        """Get number of layers."""
        if not self.heads:
            return 0
        return max(h.layer for h in self.heads) + 1
    
    @property
    def num_heads_per_layer(self) -> int:
        """Get number of heads per layer."""
        if not self.heads:
            return 0
        layer_0_heads = [h for h in self.heads if h.layer == 0]
        return len(layer_0_heads)
    
    def get_layer_attention(self, layer: int) -> List[AttentionHead]:
        """Get all attention heads from a specific layer."""
        return [h for h in self.heads if h.layer == layer]
    
    def get_head_attention(self, layer: int, head: int) -> Optional[AttentionHead]:
        """Get a specific attention head."""
        for h in self.heads:
            if h.layer == layer and h.head == head:
                return h
        return None
    
    def average_attention(self, layer: Optional[int] = None) -> np.ndarray:
        """
        Compute average attention across heads.
        
        Args:
            layer: If provided, average only heads from this layer.
                   If None, average across all heads.
        
        Returns:
            Averaged attention weights matrix.
        """
        if layer is not None:
            heads = self.get_layer_attention(layer)
        else:
            heads = self.heads
        
        if not heads:
            return np.array([])
        
        weights = np.stack([h.weights for h in heads], axis=0)
        return np.mean(weights, axis=0)


class AttentionAnalyzer:
    """
    Analyze and interpret attention patterns from transformer models.
    
    This class provides educational utilities for understanding what
    transformer models attend to when processing financial text.
    
    Example:
        >>> analyzer = AttentionAnalyzer()
        >>> # Create mock attention pattern for demonstration
        >>> pattern = analyzer.create_mock_pattern(
        ...     tokens=["stock", "price", "increased", "sharply"],
        ...     num_layers=2,
        ...     num_heads=4
        ... )
        >>> stats = analyzer.compute_attention_stats(pattern)
    """
    
    def __init__(self):
        """Initialize the attention analyzer."""
        pass
    
    def create_mock_pattern(
        self,
        tokens: List[str],
        num_layers: int = 2,
        num_heads: int = 4,
        seed: Optional[int] = None
    ) -> AttentionPattern:
        """
        Create a mock attention pattern for educational demonstrations.
        
        Args:
            tokens: List of token strings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            seed: Random seed for reproducibility
        
        Returns:
            AttentionPattern with mock attention weights
        """
        if seed is not None:
            np.random.seed(seed)
        
        seq_len = len(tokens)
        heads = []
        
        for layer in range(num_layers):
            for head in range(num_heads):
                # Generate random attention weights
                raw_weights = np.random.exponential(1.0, (seq_len, seq_len))
                
                # Apply softmax normalization (each row sums to 1)
                weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)
                
                heads.append(AttentionHead(
                    layer=layer,
                    head=head,
                    weights=weights
                ))
        
        return AttentionPattern(tokens=tokens, heads=heads)
    
    def compute_attention_stats(
        self,
        pattern: AttentionPattern
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute statistics about attention patterns.
        
        Args:
            pattern: AttentionPattern to analyze
        
        Returns:
            Dictionary with attention statistics
        """
        if not pattern.heads:
            return {"error": "No attention heads found"}
        
        # Compute average attention
        avg_attention = pattern.average_attention()
        
        # Entropy of attention (how focused vs. distributed)
        def attention_entropy(weights: np.ndarray) -> float:
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            # Mean entropy across all positions
            entropy = -np.sum(weights * np.log(weights + eps), axis=1)
            return float(np.mean(entropy))
        
        # Compute statistics for each head
        head_entropies = []
        for head in pattern.heads:
            head_entropies.append(attention_entropy(head.weights))
        
        # Token importance: how much attention each token receives on average
        token_importance = np.mean(avg_attention, axis=0)
        
        return {
            "num_layers": pattern.num_layers,
            "num_heads": pattern.num_heads_per_layer,
            "seq_length": len(pattern.tokens),
            "mean_entropy": float(np.mean(head_entropies)),
            "min_entropy": float(np.min(head_entropies)),
            "max_entropy": float(np.max(head_entropies)),
            "token_importance": token_importance,
            "most_attended_token": pattern.tokens[int(np.argmax(token_importance))],
            "least_attended_token": pattern.tokens[int(np.argmin(token_importance))],
        }
    
    def find_key_attention_pairs(
        self,
        pattern: AttentionPattern,
        threshold: float = 0.3,
        layer: Optional[int] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Find token pairs with high attention weights.
        
        Args:
            pattern: AttentionPattern to analyze
            threshold: Minimum attention weight to include
            layer: Specific layer to analyze (None for average)
        
        Returns:
            List of (from_token, to_token, weight) tuples
        """
        avg_attention = pattern.average_attention(layer)
        
        if avg_attention.size == 0:
            return []
        
        pairs = []
        for i, from_token in enumerate(pattern.tokens):
            for j, to_token in enumerate(pattern.tokens):
                weight = avg_attention[i, j]
                if weight >= threshold:
                    pairs.append((from_token, to_token, float(weight)))
        
        # Sort by weight descending
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs
    
    def attention_to_text_heatmap(
        self,
        pattern: AttentionPattern,
        layer: Optional[int] = None,
        head: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Generate data for a text attention heatmap visualization.
        
        Args:
            pattern: AttentionPattern to visualize
            layer: Specific layer (None for average across layers)
            head: Specific head (None for average across heads)
        
        Returns:
            Dictionary with heatmap data suitable for plotting
        """
        if head is not None and layer is not None:
            attention_head = pattern.get_head_attention(layer, head)
            if attention_head is None:
                return {"error": f"Head {head} at layer {layer} not found"}
            weights = attention_head.weights
        elif layer is not None:
            weights = pattern.average_attention(layer)
        else:
            weights = pattern.average_attention()
        
        return {
            "tokens": pattern.tokens,
            "attention_matrix": weights.tolist(),
            "layer": layer,
            "head": head,
            "shape": weights.shape,
        }


class FinancialAttentionInterpreter:
    """
    Specialized attention interpreter for financial text.
    
    Provides domain-specific analysis of attention patterns in
    financial context, identifying focus on key entities like
    companies, metrics, and sentiment words.
    
    Example:
        >>> interpreter = FinancialAttentionInterpreter()
        >>> tokens = ["Apple", "revenue", "increased", "15", "%"]
        >>> interpretation = interpreter.interpret_financial_attention(
        ...     tokens=tokens,
        ...     attention_weights=np.random.rand(5, 5)
        ... )
    """
    
    # Common financial terms that often receive high attention
    FINANCIAL_KEYWORDS = {
        "metrics": ["revenue", "profit", "earnings", "eps", "ebitda", "margin", 
                   "growth", "sales", "income", "loss", "debt", "cash"],
        "sentiment": ["increased", "decreased", "beat", "missed", "exceeded",
                     "disappointed", "strong", "weak", "bullish", "bearish"],
        "entities": ["stock", "share", "company", "market", "sector", "industry"],
        "temporal": ["q1", "q2", "q3", "q4", "quarter", "year", "annual", "fy"],
    }
    
    def __init__(self):
        """Initialize the financial attention interpreter."""
        pass
    
    def categorize_tokens(
        self,
        tokens: List[str]
    ) -> Dict[str, List[Tuple[int, str]]]:
        """
        Categorize tokens into financial categories.
        
        Args:
            tokens: List of token strings
        
        Returns:
            Dictionary mapping categories to (index, token) pairs
        """
        categories: Dict[str, List[Tuple[int, str]]] = {
            "metrics": [],
            "sentiment": [],
            "entities": [],
            "temporal": [],
            "numeric": [],
            "other": [],
        }
        
        for idx, token in enumerate(tokens):
            token_lower = token.lower().strip()
            
            # Check if numeric
            try:
                float(token_lower.replace("%", "").replace(",", ""))
                categories["numeric"].append((idx, token))
                continue
            except ValueError:
                pass
            
            # Check against keyword lists
            found = False
            for category, keywords in self.FINANCIAL_KEYWORDS.items():
                if token_lower in keywords:
                    categories[category].append((idx, token))
                    found = True
                    break
            
            if not found:
                categories["other"].append((idx, token))
        
        return categories
    
    def interpret_financial_attention(
        self,
        tokens: List[str],
        attention_weights: np.ndarray
    ) -> Dict[str, any]:
        """
        Interpret attention weights in financial context.
        
        Args:
            tokens: List of token strings
            attention_weights: Attention matrix (seq_len x seq_len)
        
        Returns:
            Dictionary with financial interpretation
        """
        # Categorize tokens
        categories = self.categorize_tokens(tokens)
        
        # Compute attention received by each category
        # (average attention TO tokens in each category)
        attention_per_token = np.mean(attention_weights, axis=0)
        
        category_attention = {}
        for category, token_pairs in categories.items():
            if token_pairs:
                indices = [idx for idx, _ in token_pairs]
                category_attention[category] = float(
                    np.mean(attention_per_token[indices])
                )
            else:
                category_attention[category] = 0.0
        
        # Find most important tokens
        top_indices = np.argsort(attention_per_token)[-3:][::-1]
        top_tokens = [(tokens[i], float(attention_per_token[i])) for i in top_indices]
        
        return {
            "token_categories": {
                cat: [t for _, t in pairs] 
                for cat, pairs in categories.items() if pairs
            },
            "attention_by_category": category_attention,
            "top_attended_tokens": top_tokens,
            "interpretation": self._generate_interpretation(
                category_attention, top_tokens
            ),
        }
    
    def _generate_interpretation(
        self,
        category_attention: Dict[str, float],
        top_tokens: List[Tuple[str, float]]
    ) -> str:
        """Generate human-readable interpretation of attention pattern."""
        lines = []
        
        # Find highest attention category
        if category_attention:
            sorted_cats = sorted(
                category_attention.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            top_cat, top_val = sorted_cats[0]
            
            if top_cat == "metrics":
                lines.append(
                    "Model focuses heavily on financial metrics."
                )
            elif top_cat == "sentiment":
                lines.append(
                    "Model pays strong attention to sentiment-bearing words."
                )
            elif top_cat == "numeric":
                lines.append(
                    "Model emphasizes numerical values in the text."
                )
            elif top_cat == "entities":
                lines.append(
                    "Model focuses on entity mentions (companies, markets)."
                )
        
        # Add top token info
        if top_tokens:
            top_token, top_weight = top_tokens[0]
            lines.append(
                f"Highest attention token: '{top_token}' "
                f"(weight: {top_weight:.3f})"
            )
        
        return " ".join(lines) if lines else "No clear attention pattern detected."


# Convenience functions for quick analysis
def analyze_attention(
    tokens: List[str],
    attention_weights: np.ndarray,
    is_financial: bool = True
) -> Dict[str, any]:
    """
    Quick attention analysis with sensible defaults.
    
    Args:
        tokens: List of token strings
        attention_weights: Attention matrix (seq_len x seq_len)
        is_financial: Whether to use financial-specific interpretation
    
    Returns:
        Dictionary with analysis results
    """
    # Create pattern from raw weights
    heads = [AttentionHead(layer=0, head=0, weights=attention_weights)]
    pattern = AttentionPattern(tokens=tokens, heads=heads)
    
    # Basic analysis
    analyzer = AttentionAnalyzer()
    stats = analyzer.compute_attention_stats(pattern)
    key_pairs = analyzer.find_key_attention_pairs(pattern, threshold=0.2)
    
    result = {
        "stats": stats,
        "key_pairs": key_pairs[:10],  # Top 10 pairs
    }
    
    # Add financial interpretation if requested
    if is_financial:
        interpreter = FinancialAttentionInterpreter()
        result["financial"] = interpreter.interpret_financial_attention(
            tokens, attention_weights
        )
    
    return result
