"""
ONNX Model Export Utilities.

This module provides utilities for exporting trained models to ONNX format
for deployment in production environments.

ONNX (Open Neural Network Exchange) is an open format for representing
machine learning models that enables interoperability between frameworks.

⚠️ EDUCATIONAL USE ONLY - Not for production trading.

Benefits of ONNX export:
- Framework-agnostic deployment
- Optimized inference performance
- Hardware acceleration support (GPU, TPU, etc.)
- Smaller model size for edge deployment
- Consistent behavior across platforms

Example:
    >>> from jsf.ml.export import ONNXExporter
    >>> 
    >>> # Export a scikit-learn model
    >>> exporter = ONNXExporter()
    >>> exporter.export_sklearn(model, "model.onnx", input_shape=(1, 10))
    >>> 
    >>> # Export a TensorFlow/Keras model
    >>> exporter.export_keras(keras_model, "model.onnx")
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import hashlib
import numpy as np


@dataclass
class ModelMetadata:
    """Metadata about an exported model."""
    
    model_name: str
    model_type: str
    export_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    input_dtype: str = "float32"
    output_dtype: str = "float32"
    framework: str = "unknown"
    version: str = "1.0.0"
    description: str = ""
    feature_names: List[str] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "export_timestamp": self.export_timestamp,
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "output_shape": list(self.output_shape) if self.output_shape else None,
            "input_dtype": self.input_dtype,
            "output_dtype": self.output_dtype,
            "framework": self.framework,
            "version": self.version,
            "description": self.description,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "custom_metadata": self.custom_metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary."""
        input_shape = tuple(data["input_shape"]) if data.get("input_shape") else None
        output_shape = tuple(data["output_shape"]) if data.get("output_shape") else None
        
        return cls(
            model_name=data.get("model_name", "unknown"),
            model_type=data.get("model_type", "unknown"),
            export_timestamp=data.get("export_timestamp", ""),
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=data.get("input_dtype", "float32"),
            output_dtype=data.get("output_dtype", "float32"),
            framework=data.get("framework", "unknown"),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            feature_names=data.get("feature_names", []),
            target_names=data.get("target_names", []),
            custom_metadata=data.get("custom_metadata", {}),
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save metadata to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelMetadata":
        """Load metadata from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def _check_onnx():
    """Check if ONNX is available."""
    try:
        import onnx
        return onnx
    except ImportError:
        raise ImportError(
            "ONNX is not installed.\n"
            "Install with: pip install onnx onnxruntime"
        )


def _check_skl2onnx():
    """Check if skl2onnx is available for sklearn export."""
    try:
        import skl2onnx
        return skl2onnx
    except ImportError:
        raise ImportError(
            "skl2onnx is not installed.\n"
            "Install with: pip install skl2onnx"
        )


def _check_tf2onnx():
    """Check if tf2onnx is available for TensorFlow export."""
    try:
        import tf2onnx
        return tf2onnx
    except ImportError:
        raise ImportError(
            "tf2onnx is not installed.\n"
            "Install with: pip install tf2onnx"
        )


class ONNXExporter:
    """
    Export machine learning models to ONNX format.
    
    Supports exporting models from:
    - scikit-learn (RandomForest, XGBoost, LightGBM, etc.)
    - TensorFlow/Keras (Sequential, Functional API)
    - PyTorch (with tracing)
    
    Example:
        >>> exporter = ONNXExporter()
        >>> 
        >>> # For sklearn models
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> model.fit(X_train, y_train)
        >>> exporter.export_sklearn(model, "rf_model.onnx", input_shape=(1, 10))
        
        >>> # For Keras models
        >>> exporter.export_keras(keras_model, "nn_model.onnx")
    """
    
    def __init__(self, opset_version: int = 13):
        """
        Initialize ONNX exporter.
        
        Args:
            opset_version: ONNX opset version (default: 13)
        """
        self.opset_version = opset_version
    
    def export_sklearn(
        self,
        model: Any,
        output_path: Union[str, Path],
        input_shape: Tuple[int, ...],
        input_dtype: str = "float32",
        model_name: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
        save_metadata: bool = True,
    ) -> ModelMetadata:
        """
        Export a scikit-learn model to ONNX format.
        
        Args:
            model: Trained sklearn model (or XGBoost/LightGBM)
            output_path: Path for output ONNX file
            input_shape: Expected input shape (batch_size, n_features)
            input_dtype: Input data type (default: float32)
            model_name: Name for the model
            feature_names: List of feature names
            target_names: List of target/class names
            save_metadata: Whether to save metadata JSON alongside
        
        Returns:
            ModelMetadata object with export information
        
        Note:
            Requires skl2onnx: pip install skl2onnx
        """
        skl2onnx = _check_skl2onnx()
        from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine model type
        model_class = type(model).__name__
        
        # Create input type specification
        if input_dtype == "float32":
            initial_type = [("input", FloatTensorType(list(input_shape)))]
        else:
            initial_type = [("input", DoubleTensorType(list(input_shape)))]
        
        # Convert to ONNX
        onnx_model = skl2onnx.convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=self.opset_version,
        )
        
        # Save model
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        # Create metadata
        metadata = ModelMetadata(
            model_name=model_name or model_class,
            model_type=model_class,
            input_shape=input_shape,
            input_dtype=input_dtype,
            framework="sklearn",
            feature_names=feature_names or [],
            target_names=target_names or [],
        )
        
        if save_metadata:
            metadata_path = output_path.with_suffix('.json')
            metadata.save(metadata_path)
        
        return metadata
    
    def export_keras(
        self,
        model: Any,
        output_path: Union[str, Path],
        model_name: Optional[str] = None,
        input_signature: Optional[Any] = None,
        save_metadata: bool = True,
    ) -> ModelMetadata:
        """
        Export a Keras/TensorFlow model to ONNX format.
        
        Args:
            model: Trained Keras model
            output_path: Path for output ONNX file
            model_name: Name for the model
            input_signature: TensorFlow input signature (optional)
            save_metadata: Whether to save metadata JSON alongside
        
        Returns:
            ModelMetadata object with export information
        
        Note:
            Requires tf2onnx: pip install tf2onnx
        """
        tf2onnx = _check_tf2onnx()
        import tensorflow as tf
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get model info
        model_class = type(model).__name__
        
        # Get input/output shapes
        input_shape = None
        output_shape = None
        
        if hasattr(model, 'input_shape'):
            input_shape = model.input_shape
        if hasattr(model, 'output_shape'):
            output_shape = model.output_shape
        
        # Convert to ONNX
        if input_signature is not None:
            spec = input_signature
        elif hasattr(model, 'input_spec') and model.input_spec:
            spec = model.input_spec
        else:
            # Create default spec from input shape
            if input_shape:
                spec = [tf.TensorSpec(input_shape, tf.float32, name='input')]
            else:
                spec = None
        
        # Use tf2onnx.convert for conversion
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            opset=self.opset_version,
            output_path=str(output_path),
        )
        
        # Create metadata
        metadata = ModelMetadata(
            model_name=model_name or model_class,
            model_type=model_class,
            input_shape=tuple(input_shape) if input_shape else None,
            output_shape=tuple(output_shape) if output_shape else None,
            framework="tensorflow/keras",
        )
        
        if save_metadata:
            metadata_path = output_path.with_suffix('.json')
            metadata.save(metadata_path)
        
        return metadata
    
    def validate_onnx(self, model_path: Union[str, Path]) -> bool:
        """
        Validate an ONNX model file.
        
        Args:
            model_path: Path to ONNX model file
        
        Returns:
            True if model is valid, raises exception otherwise
        """
        onnx = _check_onnx()
        
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        return True
    
    def get_model_info(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about an ONNX model.
        
        Args:
            model_path: Path to ONNX model file
        
        Returns:
            Dictionary with model information
        """
        onnx = _check_onnx()
        
        model = onnx.load(str(model_path))
        
        info = {
            "ir_version": model.ir_version,
            "opset_version": model.opset_import[0].version if model.opset_import else None,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "graph_name": model.graph.name,
            "inputs": [],
            "outputs": [],
        }
        
        # Get input info
        for inp in model.graph.input:
            input_info = {
                "name": inp.name,
                "shape": [dim.dim_value for dim in inp.type.tensor_type.shape.dim],
                "type": inp.type.tensor_type.elem_type,
            }
            info["inputs"].append(input_info)
        
        # Get output info
        for out in model.graph.output:
            output_info = {
                "name": out.name,
                "shape": [dim.dim_value for dim in out.type.tensor_type.shape.dim],
                "type": out.type.tensor_type.elem_type,
            }
            info["outputs"].append(output_info)
        
        return info


class MockONNXExporter:
    """
    Mock ONNX exporter for educational demonstrations.
    
    This class simulates ONNX export functionality without requiring
    the actual ONNX libraries installed. Useful for understanding
    the export workflow.
    
    Example:
        >>> exporter = MockONNXExporter()
        >>> metadata = exporter.mock_export(
        ...     model_name="TradingModel",
        ...     model_type="RandomForest",
        ...     input_shape=(1, 50),
        ...     output_path="model.onnx"
        ... )
        >>> print(metadata.to_dict())
    """
    
    def mock_export(
        self,
        model_name: str,
        model_type: str,
        input_shape: Tuple[int, ...],
        output_path: Union[str, Path],
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
        description: str = "",
    ) -> ModelMetadata:
        """
        Simulate model export and create metadata.
        
        Args:
            model_name: Name for the model
            model_type: Type of model (e.g., "RandomForest")
            input_shape: Expected input shape
            output_path: Path where model would be saved
            feature_names: List of feature names
            target_names: List of target names
            description: Model description
        
        Returns:
            ModelMetadata with simulated export information
        """
        output_path = Path(output_path)
        
        # Create mock ONNX file content
        mock_content = {
            "type": "mock_onnx_model",
            "model_name": model_name,
            "model_type": model_type,
            "input_shape": list(input_shape),
            "timestamp": datetime.now().isoformat(),
            "note": "This is a mock ONNX file for educational purposes",
        }
        
        # Save mock file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(mock_content, f, indent=2)
        
        # Create and save metadata
        metadata = ModelMetadata(
            model_name=model_name,
            model_type=model_type,
            input_shape=input_shape,
            output_shape=(input_shape[0], 1),  # Assume single output
            framework="mock",
            description=description or "Mock ONNX export for educational purposes",
            feature_names=feature_names or [],
            target_names=target_names or [],
        )
        
        metadata_path = output_path.with_suffix('.json')
        metadata.save(metadata_path)
        
        return metadata
    
    def mock_inference(
        self,
        metadata: ModelMetadata,
        input_data: np.ndarray
    ) -> np.ndarray:
        """
        Simulate ONNX inference.
        
        Args:
            metadata: Model metadata
            input_data: Input array
        
        Returns:
            Mock predictions
        """
        batch_size = input_data.shape[0]
        
        # Generate deterministic mock output based on input hash
        input_hash = hashlib.md5(input_data.tobytes()).hexdigest()
        seed = int(input_hash[:8], 16) % (2**32)
        np.random.seed(seed)
        
        # Generate mock predictions
        if metadata.output_shape:
            output_shape = (batch_size,) + metadata.output_shape[1:]
        else:
            output_shape = (batch_size, 1)
        
        predictions = np.random.randn(*output_shape).astype(np.float32)
        
        return predictions


def create_exporter(mock: bool = False) -> Union[ONNXExporter, MockONNXExporter]:
    """
    Create an ONNX exporter.
    
    Args:
        mock: If True, return MockONNXExporter (no dependencies required)
    
    Returns:
        ONNXExporter or MockONNXExporter instance
    """
    if mock:
        return MockONNXExporter()
    
    return ONNXExporter()


# Export model checksum utilities
def compute_model_checksum(model_path: Union[str, Path]) -> str:
    """
    Compute SHA256 checksum of a model file.
    
    Args:
        model_path: Path to model file
    
    Returns:
        SHA256 hex digest
    """
    path = Path(model_path)
    sha256_hash = hashlib.sha256()
    
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


def verify_model_checksum(
    model_path: Union[str, Path],
    expected_checksum: str
) -> bool:
    """
    Verify model file checksum matches expected value.
    
    Args:
        model_path: Path to model file
        expected_checksum: Expected SHA256 hex digest
    
    Returns:
        True if checksum matches, False otherwise
    """
    actual_checksum = compute_model_checksum(model_path)
    return actual_checksum == expected_checksum
