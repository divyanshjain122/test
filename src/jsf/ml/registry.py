"""
Model Registry for Tracking and Versioning Trained Models.

This module provides utilities for managing, versioning, and tracking
trained machine learning models in a local registry.

⚠️ EDUCATIONAL USE ONLY - Not for production trading.

Features:
- Local model storage with versioning
- Model metadata and performance tracking
- Model comparison utilities
- Experiment tracking integration

Example:
    >>> from jsf.ml.registry import ModelRegistry
    >>> 
    >>> # Create registry
    >>> registry = ModelRegistry("./models")
    >>> 
    >>> # Register a model
    >>> registry.register(
    ...     model=trained_model,
    ...     name="momentum_classifier",
    ...     version="1.0.0",
    ...     metrics={"accuracy": 0.85, "sharpe": 1.5}
    ... )
    >>> 
    >>> # Load a model
    >>> model = registry.load("momentum_classifier", version="latest")
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import shutil
import pickle
import hashlib
from enum import Enum


class ModelStatus(Enum):
    """Model lifecycle status."""
    
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


@dataclass
class ModelVersion:
    """Information about a specific model version."""
    
    version: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    model_path: str = ""
    metadata_path: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    status: ModelStatus = ModelStatus.DEVELOPMENT
    description: str = ""
    tags: List[str] = field(default_factory=list)
    checksum: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "model_path": self.model_path,
            "metadata_path": self.metadata_path,
            "metrics": self.metrics,
            "params": self.params,
            "status": self.status.value,
            "description": self.description,
            "tags": self.tags,
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        return cls(
            version=data["version"],
            created_at=data.get("created_at", ""),
            model_path=data.get("model_path", ""),
            metadata_path=data.get("metadata_path", ""),
            metrics=data.get("metrics", {}),
            params=data.get("params", {}),
            status=ModelStatus(data.get("status", "development")),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            checksum=data.get("checksum", ""),
        )


@dataclass
class RegisteredModel:
    """A registered model with multiple versions."""
    
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    versions: Dict[str, ModelVersion] = field(default_factory=dict)
    latest_version: str = ""
    production_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "versions": {k: v.to_dict() for k, v in self.versions.items()},
            "latest_version": self.latest_version,
            "production_version": self.production_version,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegisteredModel":
        """Create from dictionary."""
        versions = {
            k: ModelVersion.from_dict(v)
            for k, v in data.get("versions", {}).items()
        }
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            versions=versions,
            latest_version=data.get("latest_version", ""),
            production_version=data.get("production_version"),
            tags=data.get("tags", []),
        )
    
    def get_version(self, version: str = "latest") -> Optional[ModelVersion]:
        """Get a specific version or latest."""
        if version == "latest":
            version = self.latest_version
        return self.versions.get(version)
    
    def list_versions(self) -> List[str]:
        """List all version strings."""
        return list(self.versions.keys())


class ModelRegistry:
    """
    Local model registry for tracking and versioning trained models.
    
    Provides a simple file-based registry for storing and retrieving
    trained models with version control and metadata.
    
    Example:
        >>> registry = ModelRegistry("./model_registry")
        >>> 
        >>> # Register a new model
        >>> registry.register(
        ...     model=trained_rf,
        ...     name="price_predictor",
        ...     version="1.0.0",
        ...     metrics={"mse": 0.05, "r2": 0.85}
        ... )
        >>> 
        >>> # List all models
        >>> models = registry.list_models()
        >>> 
        >>> # Load a model
        >>> model = registry.load("price_predictor")
        >>> 
        >>> # Compare versions
        >>> comparison = registry.compare_versions(
        ...     "price_predictor", ["1.0.0", "1.1.0"]
        ... )
    """
    
    def __init__(self, registry_path: Union[str, Path] = "./model_registry"):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Path to registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self._index_path = self.registry_path / "registry_index.json"
        self._models: Dict[str, RegisteredModel] = {}
        
        self._load_index()
    
    def _load_index(self) -> None:
        """Load registry index from disk."""
        if self._index_path.exists():
            with open(self._index_path, 'r') as f:
                data = json.load(f)
                self._models = {
                    name: RegisteredModel.from_dict(model_data)
                    for name, model_data in data.get("models", {}).items()
                }
    
    def _save_index(self) -> None:
        """Save registry index to disk."""
        data = {
            "models": {
                name: model.to_dict()
                for name, model in self._models.items()
            },
            "updated_at": datetime.now().isoformat(),
        }
        with open(self._index_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _compute_checksum(self, model_path: Path) -> str:
        """Compute SHA256 checksum of model file."""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def register(
        self,
        model: Any,
        name: str,
        version: str,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        status: ModelStatus = ModelStatus.DEVELOPMENT,
    ) -> ModelVersion:
        """
        Register a trained model in the registry.
        
        Args:
            model: Trained model object (must be picklable)
            name: Model name (unique identifier)
            version: Version string (e.g., "1.0.0")
            metrics: Performance metrics dict
            params: Model parameters/config dict
            description: Version description
            tags: List of tags for categorization
            status: Model lifecycle status
        
        Returns:
            ModelVersion object for registered model
        
        Raises:
            ValueError: If version already exists
        """
        # Get or create registered model entry
        if name not in self._models:
            self._models[name] = RegisteredModel(name=name)
        
        registered_model = self._models[name]
        
        # Check version doesn't exist
        if version in registered_model.versions:
            raise ValueError(f"Version {version} already exists for model {name}")
        
        # Create model directory
        model_dir = self.registry_path / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Compute checksum
        checksum = self._compute_checksum(model_path)
        
        # Save metadata
        metadata_path = model_dir / "metadata.json"
        metadata = {
            "name": name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics or {},
            "params": params or {},
            "description": description,
            "tags": tags or [],
            "checksum": checksum,
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create version entry
        model_version = ModelVersion(
            version=version,
            model_path=str(model_path),
            metadata_path=str(metadata_path),
            metrics=metrics or {},
            params=params or {},
            status=status,
            description=description,
            tags=tags or [],
            checksum=checksum,
        )
        
        # Update registered model
        registered_model.versions[version] = model_version
        registered_model.latest_version = version
        registered_model.updated_at = datetime.now().isoformat()
        
        # Save index
        self._save_index()
        
        return model_version
    
    def load(
        self,
        name: str,
        version: str = "latest"
    ) -> Any:
        """
        Load a model from the registry.
        
        Args:
            name: Model name
            version: Version string or "latest"
        
        Returns:
            Loaded model object
        
        Raises:
            KeyError: If model or version not found
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")
        
        registered_model = self._models[name]
        model_version = registered_model.get_version(version)
        
        if model_version is None:
            raise KeyError(f"Version '{version}' not found for model '{name}'")
        
        # Load and return model
        with open(model_version.model_path, 'rb') as f:
            return pickle.load(f)
    
    def get_model_info(self, name: str) -> Optional[RegisteredModel]:
        """
        Get information about a registered model.
        
        Args:
            name: Model name
        
        Returns:
            RegisteredModel or None if not found
        """
        return self._models.get(name)
    
    def get_version_info(
        self,
        name: str,
        version: str = "latest"
    ) -> Optional[ModelVersion]:
        """
        Get information about a specific version.
        
        Args:
            name: Model name
            version: Version string or "latest"
        
        Returns:
            ModelVersion or None if not found
        """
        if name not in self._models:
            return None
        return self._models[name].get_version(version)
    
    def list_models(self) -> List[str]:
        """
        List all registered model names.
        
        Returns:
            List of model names
        """
        return list(self._models.keys())
    
    def list_versions(self, name: str) -> List[str]:
        """
        List all versions of a model.
        
        Args:
            name: Model name
        
        Returns:
            List of version strings
        """
        if name not in self._models:
            return []
        return self._models[name].list_versions()
    
    def compare_versions(
        self,
        name: str,
        versions: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare metrics across model versions.
        
        Args:
            name: Model name
            versions: List of version strings to compare
        
        Returns:
            Dictionary mapping version to metrics
        """
        if name not in self._models:
            return {}
        
        registered_model = self._models[name]
        comparison = {}
        
        for version in versions:
            model_version = registered_model.versions.get(version)
            if model_version:
                comparison[version] = {
                    "metrics": model_version.metrics,
                    "created_at": model_version.created_at,
                    "status": model_version.status.value,
                }
        
        return comparison
    
    def set_production_version(
        self,
        name: str,
        version: str
    ) -> None:
        """
        Set the production version for a model.
        
        Args:
            name: Model name
            version: Version to mark as production
        
        Raises:
            KeyError: If model or version not found
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found")
        
        registered_model = self._models[name]
        
        if version not in registered_model.versions:
            raise KeyError(f"Version '{version}' not found")
        
        # Update statuses
        if registered_model.production_version:
            old_version = registered_model.versions.get(
                registered_model.production_version
            )
            if old_version:
                old_version.status = ModelStatus.ARCHIVED
        
        registered_model.production_version = version
        registered_model.versions[version].status = ModelStatus.PRODUCTION
        registered_model.updated_at = datetime.now().isoformat()
        
        self._save_index()
    
    def delete_version(
        self,
        name: str,
        version: str,
        force: bool = False
    ) -> bool:
        """
        Delete a model version.
        
        Args:
            name: Model name
            version: Version to delete
            force: Force delete even if production
        
        Returns:
            True if deleted, False if not found
        
        Raises:
            ValueError: If trying to delete production without force
        """
        if name not in self._models:
            return False
        
        registered_model = self._models[name]
        
        if version not in registered_model.versions:
            return False
        
        # Check if production
        if registered_model.production_version == version and not force:
            raise ValueError(
                f"Cannot delete production version. Use force=True to override."
            )
        
        # Get version info for path
        model_version = registered_model.versions[version]
        
        # Delete files
        model_dir = Path(model_version.model_path).parent
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remove from registry
        del registered_model.versions[version]
        
        # Update latest
        if registered_model.latest_version == version:
            versions = list(registered_model.versions.keys())
            registered_model.latest_version = versions[-1] if versions else ""
        
        # Update production
        if registered_model.production_version == version:
            registered_model.production_version = None
        
        registered_model.updated_at = datetime.now().isoformat()
        
        self._save_index()
        return True
    
    def search(
        self,
        tags: Optional[List[str]] = None,
        status: Optional[ModelStatus] = None,
        metric_filter: Optional[Dict[str, tuple]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search models by criteria.
        
        Args:
            tags: Filter by tags (any match)
            status: Filter by status
            metric_filter: Filter by metrics, e.g., {"accuracy": (0.8, None)}
                           means accuracy >= 0.8
        
        Returns:
            List of matching model/version info dicts
        """
        results = []
        
        for name, model in self._models.items():
            for version_str, version in model.versions.items():
                # Tag filter
                if tags:
                    if not any(t in version.tags for t in tags):
                        continue
                
                # Status filter
                if status and version.status != status:
                    continue
                
                # Metric filter
                if metric_filter:
                    match = True
                    for metric_name, (min_val, max_val) in metric_filter.items():
                        metric_val = version.metrics.get(metric_name)
                        if metric_val is None:
                            match = False
                            break
                        if min_val is not None and metric_val < min_val:
                            match = False
                            break
                        if max_val is not None and metric_val > max_val:
                            match = False
                            break
                    if not match:
                        continue
                
                results.append({
                    "model_name": name,
                    "version": version_str,
                    "metrics": version.metrics,
                    "status": version.status.value,
                    "created_at": version.created_at,
                })
        
        return results


class ExperimentTracker:
    """
    Simple experiment tracking for model training runs.
    
    Tracks training experiments with parameters, metrics, and artifacts.
    Integrates with ModelRegistry for model storage.
    
    Example:
        >>> tracker = ExperimentTracker("./experiments")
        >>> 
        >>> with tracker.start_run("momentum_experiment"):
        ...     tracker.log_params({"n_estimators": 100, "max_depth": 5})
        ...     # ... training ...
        ...     tracker.log_metrics({"train_acc": 0.9, "val_acc": 0.85})
        ...     tracker.log_artifact("feature_importance.png")
    """
    
    def __init__(
        self,
        experiments_path: Union[str, Path] = "./experiments",
        registry: Optional[ModelRegistry] = None
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiments_path: Path to store experiments
            registry: Optional ModelRegistry for model storage
        """
        self.experiments_path = Path(experiments_path)
        self.experiments_path.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        
        self._current_run: Optional[Dict[str, Any]] = None
        self._runs: List[Dict[str, Any]] = []
        
        self._load_runs()
    
    def _load_runs(self) -> None:
        """Load existing runs from disk."""
        index_path = self.experiments_path / "runs_index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                self._runs = json.load(f).get("runs", [])
    
    def _save_runs(self) -> None:
        """Save runs to disk."""
        index_path = self.experiments_path / "runs_index.json"
        with open(index_path, 'w') as f:
            json.dump({"runs": self._runs}, f, indent=2)
    
    def start_run(self, name: str) -> "ExperimentTracker":
        """
        Start a new experiment run.
        
        Args:
            name: Experiment name
        
        Returns:
            self for context manager usage
        """
        run_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self._current_run = {
            "run_id": run_id,
            "name": name,
            "started_at": datetime.now().isoformat(),
            "ended_at": None,
            "params": {},
            "metrics": {},
            "artifacts": [],
            "status": "running",
        }
        
        # Create run directory
        run_dir = self.experiments_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run(status="failed" if exc_type else "completed")
    
    def end_run(self, status: str = "completed") -> None:
        """End the current run."""
        if self._current_run:
            self._current_run["ended_at"] = datetime.now().isoformat()
            self._current_run["status"] = status
            self._runs.append(self._current_run)
            self._save_runs()
            self._current_run = None
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters for current run."""
        if self._current_run:
            self._current_run["params"].update(params)
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics for current run."""
        if self._current_run:
            self._current_run["metrics"].update(metrics)
    
    def log_artifact(self, path: Union[str, Path]) -> None:
        """Log an artifact path for current run."""
        if self._current_run:
            self._current_run["artifacts"].append(str(path))
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific run by ID."""
        for run in self._runs:
            if run["run_id"] == run_id:
                return run
        return None
    
    def list_runs(
        self,
        name_filter: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List experiment runs.
        
        Args:
            name_filter: Filter by experiment name
            limit: Maximum runs to return
        
        Returns:
            List of run info dicts
        """
        runs = self._runs
        
        if name_filter:
            runs = [r for r in runs if name_filter in r["name"]]
        
        # Sort by started_at descending
        runs = sorted(runs, key=lambda r: r["started_at"], reverse=True)
        
        return runs[:limit]
    
    def get_best_run(
        self,
        name: str,
        metric: str,
        maximize: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best run for an experiment by metric.
        
        Args:
            name: Experiment name
            metric: Metric to optimize
            maximize: True to find max, False for min
        
        Returns:
            Best run or None
        """
        runs = [r for r in self._runs if r["name"] == name]
        runs = [r for r in runs if metric in r["metrics"]]
        
        if not runs:
            return None
        
        if maximize:
            return max(runs, key=lambda r: r["metrics"][metric])
        else:
            return min(runs, key=lambda r: r["metrics"][metric])
