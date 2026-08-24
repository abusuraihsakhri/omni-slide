"""OmniSlide Pro — GPU-Accelerated & Self-Healing Whole-Slide & Microscopy Suite."""

from jp2_tiff_converter.config import Config
from jp2_tiff_converter.converter import ConversionResult, JP2Converter
from jp2_tiff_converter.gui import launch_gui

__title__ = "OmniSlide"
__version__ = "2.0.0"
__all__ = ["Config", "JP2Converter", "ConversionResult", "launch_gui"]