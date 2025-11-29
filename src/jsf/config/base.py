"""
Base configuration models.

Provides base classes and common utilities for all config schemas.
"""

from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path
import json

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic import Field


class JSFBaseConfig(BaseModel):
    """
    Base configuration class for all JSF configs.
    
    Provides common functionality:
    - JSON serialization/deserialization
    - Validation
    - Immutability (frozen after creation)
    """
    
    model_config = ConfigDict(
        frozen=False,  # Allow modification during construction
        validate_assignment=True,  # Validate on attribute assignment
        arbitrary_types_allowed=True,  # Allow custom types
        use_enum_values=True,  # Store enum values, not enum objects
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.
        
        Returns:
            Dictionary representation
        """
        return self.model_dump(mode="python", exclude_none=True)
    
    def to_json(self, filepath: Optional[Path] = None, **kwargs: Any) -> str:
        """
        Convert config to JSON string or save to file.
        
        Args:
            filepath: Optional path to save JSON file
            **kwargs: Additional arguments for json.dumps
            
        Returns:
            JSON string
        """
        json_str = self.model_dump_json(indent=2, exclude_none=True, **kwargs)
        
        if filepath is not None:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(json_str)
        
        return json_str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JSFBaseConfig":
        """
        Create config from dictionary.
        
        Args:
            data: Dictionary with config data
            
        Returns:
            Config instance
        """
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str_or_path: str | Path) -> "JSFBaseConfig":
        """
        Create config from JSON string or file.
        
        Args:
            json_str_or_path: JSON string or path to JSON file
            
        Returns:
            Config instance
        """
        # Check if it's a file path
        if isinstance(json_str_or_path, (str, Path)):
            path = Path(json_str_or_path)
            if path.exists() and path.is_file():
                json_str = path.read_text()
            else:
                json_str = json_str_or_path
        else:
            json_str = json_str_or_path
        
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def copy_with(self, **updates: Any) -> "JSFBaseConfig":
        """
        Create a copy with updated fields.
        
        Args:
            **updates: Fields to update
            
        Returns:
            New config instance with updates
        """
        data = self.to_dict()
        data.update(updates)
        return self.__class__.from_dict(data)
    
    def __repr__(self) -> str:
        """String representation."""
        fields = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"{self.__class__.__name__}({fields})"


class DateRangeConfig(JSFBaseConfig):
    """Configuration for date ranges."""
    
    start_date: str = Field(
        ...,
        description="Start date in YYYY-MM-DD format",
        examples=["2015-01-01"],
    )
    
    end_date: str = Field(
        ...,
        description="End date in YYYY-MM-DD format",
        examples=["2023-12-31"],
    )
    
    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}")
        return v
    
    @field_validator("end_date")
    @classmethod
    def validate_end_after_start(cls, v: str, info: Any) -> str:
        """Validate end date is after start date."""
        if "start_date" in info.data:
            start = datetime.strptime(info.data["start_date"], "%Y-%m-%d")
            end = datetime.strptime(v, "%Y-%m-%d")
            if end <= start:
                raise ValueError(
                    f"end_date ({v}) must be after start_date ({info.data['start_date']})"
                )
        return v
