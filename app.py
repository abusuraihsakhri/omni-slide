"""Standalone Desktop Application Entrypoint for JP2 to TIFF Converter Pro."""

import sys
from pathlib import Path

# Add src to sys.path if not installed as package
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from jp2_tiff_converter.config import Config
from jp2_tiff_converter.gui import launch_gui
from jp2_tiff_converter.logging_config import setup_logging

if __name__ == "__main__":
    config_file = Path("config.yaml")
    cfg = Config.from_file(config_file) if config_file.exists() else Config()
    setup_logging(cfg)
    launch_gui(cfg)
