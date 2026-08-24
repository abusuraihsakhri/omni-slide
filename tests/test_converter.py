"""Comprehensive Unit and Integration Tests for JP2 to TIFF Converter Pro."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import tifffile

from jp2_tiff_converter.config import Config
from jp2_tiff_converter.converter import (
    CUDA_AVAILABLE,
    JP2Converter,
    ConversionResult,
)


class TestConfig:
    """Test configuration handling and serialization."""

    def test_default_config(self):
        cfg = Config()
        assert cfg.compression == "tiff_deflate"
        assert cfg.tile_size == (256, 256)
        assert cfg.pyramid is False

    def test_config_roundtrip(self):
        cfg = Config(
            compression="tiff_lzw",
            tile_size=(512, 512),
            pyramid=True,
            input_dir=Path("/tmp/in"),
            output_dir=Path("/tmp/out"),
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            cfg.to_file(temp_path)
            loaded = Config.from_file(temp_path)

            assert loaded.compression == "tiff_lzw"
            assert loaded.tile_size == (512, 512)
            assert loaded.pyramid is True
            assert loaded.input_dir == Path("/tmp/in")
            assert loaded.output_dir == Path("/tmp/out")
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestConverter:
    """Test conversion logic, GPU acceleration, and self-healing engines."""

    @pytest.fixture
    def config(self):
        return Config()

    @pytest.fixture
    def converter(self, config):
        return JP2Converter(config)

    @pytest.fixture
    def sample_jp2(self):
        """Create a temporary valid JP2 file for testing."""
        arr = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        with tempfile.NamedTemporaryFile(suffix=".jp2", delete=False) as f:
            temp_path = Path(f.name)
        img.save(temp_path, format="JPEG2000")
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()

    def test_converter_initialization(self, converter):
        assert converter.config is not None

    def test_single_file_conversion(self, converter, sample_jp2):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            out_path = Path(f.name)

        try:
            res = converter.convert_file(sample_jp2, out_path, overwrite=True)
            assert res.success is True
            assert res.dimensions == (300, 200)
            assert res.channels == 3
            assert out_path.exists()
            assert out_path.stat().st_size > 0

            # Verify written TIFF can be read by tifffile
            read_back = tifffile.imread(str(out_path))
            assert read_back.shape == (200, 300, 3)
        finally:
            if out_path.exists():
                out_path.unlink()

    def test_pyramid_bigtiff_conversion(self, converter, sample_jp2):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            out_path = Path(f.name)

        try:
            res = converter.convert_file(
                sample_jp2,
                out_path,
                pyramid=True,
                compression="tiff_deflate",
                tile_size=(64, 64),
                overwrite=True,
            )
            assert res.success is True
            assert out_path.exists()
            if CUDA_AVAILABLE:
                assert res.gpu_accelerated is True

            with tifffile.TiffFile(str(out_path)) as tif:
                assert len(tif.pages) >= 1
                base_page = tif.pages[0]
                assert base_page.shape == (200, 300, 3)
        finally:
            if out_path.exists():
                out_path.unlink()

    def test_self_healing_stream_repair(self, converter):
        """Test self-healing on truncated/damaged JP2 files."""
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        with tempfile.NamedTemporaryFile(suffix=".jp2", delete=False) as f:
            temp_in = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            temp_out = Path(f.name)

        img.save(temp_in, format="JPEG2000")

        # Simulate stream truncation
        with open(temp_in, "rb") as f:
            raw = f.read()
        truncated = raw[: int(len(raw) * 0.95)]  # Cut off trailing 5%
        with open(temp_in, "wb") as f:
            f.write(truncated)

        try:
            res = converter.convert_file(temp_in, temp_out, overwrite=True)
            # Self-healing engine should recover truncated stream via Pillow/OpenJPEG load_truncated
            assert res.success is True
            assert temp_out.exists()
        finally:
            if temp_in.exists():
                temp_in.unlink()
            if temp_out.exists():
                temp_out.unlink()

    def test_batch_conversion(self, converter):
        with tempfile.TemporaryDirectory() as temp_in, tempfile.TemporaryDirectory() as temp_out:
            in_dir = Path(temp_in)
            out_dir = Path(temp_out)

            arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(in_dir / "slide1.jp2", format="JPEG2000")
            img.save(in_dir / "slide2.jpx", format="JPEG2000")

            results = converter.batch_convert(input_dir=in_dir, output_dir=out_dir)
            assert len(results) == 2
            assert all(r.success for r in results)
            assert (out_dir / "slide1.tif").exists()
            assert (out_dir / "slide2.tif").exists()

    def test_nonexistent_file_handling(self, converter):
        res = converter.convert_file(Path("does_not_exist.jp2"), Path("out.tif"))
        assert res.success is False
        assert "not exist" in res.error_message.lower() or "could not read" in res.error_message.lower()


class TestCLI:
    """Test CLI commands and interface."""

    def test_check_deps_command(self):
        from click.testing import CliRunner
        from jp2_tiff_converter.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["check-deps"])
        assert result.exit_code == 0
        assert "tifffile" in result.output

    def test_init_config_command(self):
        from click.testing import CliRunner
        from jp2_tiff_converter.cli import cli

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            result = runner.invoke(cli, ["init-config", str(config_path)])
            assert result.exit_code == 0
            assert config_path.exists()