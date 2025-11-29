"""I/O utilities for JSF-Core."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to directory

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Data to save
        filepath: Path to JSON file
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded data
    """
    with open(filepath, "r") as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: Path) -> None:
    """
    Save object to pickle file.

    Args:
        obj: Object to save
        filepath: Path to pickle file
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Path) -> Any:
    """
    Load object from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded object
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_dataframe(
    df: pd.DataFrame,
    filepath: Path,
    format: str = "parquet",
    **kwargs: Any,
) -> None:
    """
    Save DataFrame to file.

    Args:
        df: DataFrame to save
        filepath: Path to output file
        format: File format ('parquet', 'csv', 'hdf')
        **kwargs: Additional arguments for the save function
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    if format == "parquet":
        df.to_parquet(filepath, **kwargs)
    elif format == "csv":
        df.to_csv(filepath, **kwargs)
    elif format == "hdf":
        df.to_hdf(filepath, key="data", **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_dataframe(
    filepath: Path,
    format: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load DataFrame from file.

    Args:
        filepath: Path to input file
        format: File format (auto-detected from extension if None)
        **kwargs: Additional arguments for the load function

    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if format is None:
        format = filepath.suffix.lstrip(".")
    
    if format == "parquet":
        return pd.read_parquet(filepath, **kwargs)
    elif format == "csv":
        return pd.read_csv(filepath, **kwargs)
    elif format in ["hdf", "h5"]:
        return pd.read_hdf(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def file_exists(filepath: Path) -> bool:
    """
    Check if file exists.

    Args:
        filepath: Path to check

    Returns:
        True if file exists, False otherwise
    """
    return Path(filepath).exists()
