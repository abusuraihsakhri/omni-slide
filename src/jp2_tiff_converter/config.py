"""Configuration management for JP2 to TIFF converter."""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Config:
    """Application configuration."""
    # Conversion settings
    compression: str = "tiff_deflate"  # tiff_deflate, tiff_lzw, tiff_jpeg, zstd, packbits, none
    tile_size: Tuple[int, int] = (256, 256)
    pyramid: bool = False  # Generate multi-resolution pyramid BigTIFF
    prefer_glymur: bool = False  # Prefer Pillow/OpenCV for stable JP2 decode

    # Paths
    input_dir: Path = field(default_factory=lambda: Path("./input"))
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    log_dir: Path = field(default_factory=lambda: Path("./logs"))

    # Processing
    recursive: bool = True
    file_pattern: str = "*.jp2"
    overwrite: bool = False
    min_output_size_mb: float = 0.01

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True

    # Cloud upload (optional, keys retrieved from environment variables)
    upload_enabled: bool = False
    upload_provider: str = ""  # aws, azure, gcp
    upload_bucket: str = ""

    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Convert paths
        for key in ("input_dir", "output_dir", "log_dir"):
            if key in data and data[key]:
                data[key] = Path(data[key])

        if "tile_size" in data and isinstance(data["tile_size"], list):
            data["tile_size"] = tuple(data["tile_size"])

        # Filter out unknown keys safely
        valid_keys = cls.__dataclass_fields__.keys()
        clean_data = {k: v for k, v in data.items() if k in valid_keys}

        return cls(**clean_data)

    def to_file(self, config_path: Path) -> None:
        """Save configuration to YAML file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "compression": self.compression,
            "tile_size": list(self.tile_size),
            "pyramid": self.pyramid,
            "prefer_glymur": self.prefer_glymur,
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "log_dir": str(self.log_dir),
            "recursive": self.recursive,
            "file_pattern": self.file_pattern,
            "overwrite": self.overwrite,
            "min_output_size_mb": self.min_output_size_mb,
            "log_level": self.log_level,
            "log_to_file": self.log_to_file,
            "log_to_console": self.log_to_console,
            "upload_enabled": self.upload_enabled,
            "upload_provider": self.upload_provider,
            "upload_bucket": self.upload_bucket,
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def create_default(cls, config_path: Path) -> "Config":
        """Create a default configuration file."""
        config = cls()
        config.to_file(config_path)
        return config


DEFAULT_CONFIG_YAML = """# JP2 to TIFF Converter Configuration

# Conversion settings
compression: "tiff_deflate"  # Options: tiff_deflate, tiff_lzw, tiff_jpeg, zstd, packbits, none
tile_size: [256, 256]
pyramid: false  # Multi-resolution pyramidal BigTIFF
prefer_glymur: false

# Paths
input_dir: "./input"
output_dir: "./output"
log_dir: "./logs"

# Processing
recursive: true
file_pattern: "*.jp2"
overwrite: false
min_output_size_mb: 0.01

# Logging
log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
log_to_file: true
log_to_console: true

# Cloud upload (credentials loaded from environment variables)
upload_enabled: false
upload_provider: ""  # aws, azure, gcp
upload_bucket: ""
"""