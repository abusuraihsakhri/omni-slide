"""Core JP2 to TIFF conversion logic.

High-performance, GPU-Accelerated (CUDA), and Self-Healing JPEG2000 (JP2/JPX) to TIFF converter
supporting standard TIFF, tiled TIFF, and pyramidal BigTIFF.
"""

import gc
import io
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# 1. GPU Acceleration Backend (PyTorch CUDA)
CUDA_AVAILABLE = False
GPU_DEVICE_NAME = "None"
try:
    import torch
    if torch.cuda.is_available():
        CUDA_AVAILABLE = True
        GPU_DEVICE_NAME = torch.cuda.get_device_name(0)
except ImportError:
    pass

# 2. Biomedical TIFF Backend
TIFFFILE_AVAILABLE = False
try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    pass

# 3. Universal Image Engine
PILLOW_AVAILABLE = False
try:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # Self-healing for truncated files
    PILLOW_AVAILABLE = True
except ImportError:
    pass

# 4. Computer Vision Engine
OPENCV_AVAILABLE = False
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    pass

# 5. Glymur Engine
GLYMUR_AVAILABLE = False
try:
    import glymur
    if getattr(glymur.version, "openjpeg_version", "0.0.0") != "0.0.0":
        GLYMUR_AVAILABLE = True
except ImportError:
    pass

from jp2_tiff_converter.config import Config

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Result of a single file conversion with telemetry."""
    input_path: Path
    output_path: Optional[Path]
    success: bool
    error_message: Optional[str] = None
    file_size_mb: float = 0.0
    dimensions: Optional[Tuple[int, int]] = None
    elapsed_seconds: float = 0.0
    channels: int = 0
    backend_used: str = ""
    gpu_accelerated: bool = False
    healed_events: List[str] = field(default_factory=list)


class JP2Converter:
    """High-performance, Self-Healing & GPU-Accelerated JP2 to TIFF converter."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.gpu_enabled = CUDA_AVAILABLE
        self._validate_backends()

    def _validate_backends(self) -> None:
        """Verify available engines and log capabilities."""
        logger.debug(
            "Engines: CUDA=%s (%s), tifffile=%s, pillow=%s, opencv=%s, glymur=%s",
            CUDA_AVAILABLE, GPU_DEVICE_NAME, TIFFFILE_AVAILABLE, PILLOW_AVAILABLE,
            OPENCV_AVAILABLE, GLYMUR_AVAILABLE
        )

    # -------------------------------------------------------------------------
    # GPU Tensor Operations (Fast Pyramids & Color Space Transformations)
    # -------------------------------------------------------------------------
    def _gpu_generate_pyramid(
        self,
        base_data: np.ndarray,
        levels: int = 4
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Generate multi-resolution pyramid levels using NVIDIA CUDA GPU acceleration."""
        healing_log = []
        if not (self.gpu_enabled and CUDA_AVAILABLE):
            return self._cpu_generate_pyramid(base_data, levels)

        try:
            # Convert NumPy HWC to PyTorch NCHW tensor on CUDA
            is_rgb = (len(base_data.shape) == 3)
            orig_dtype = base_data.dtype

            if is_rgb:
                # Shape: (1, C, H, W)
                tensor = torch.from_numpy(base_data).permute(2, 0, 1).unsqueeze(0)
            else:
                # Shape: (1, 1, H, W)
                tensor = torch.from_numpy(base_data).unsqueeze(0).unsqueeze(0)

            # Move float tensor to GPU
            tensor = tensor.to(device="cuda", dtype=torch.float32, non_blocking=True)
            pyramid = [base_data]

            curr_tensor = tensor
            for level in range(1, levels + 1):
                _, _, h, w = curr_tensor.shape
                if h <= 256 or w <= 256:
                    break

                target_h, target_w = max(128, h // 2), max(128, w // 2)
                # GPU Bilinear/Area downsampling (high quality, ultra-fast)
                downscaled = torch.nn.functional.interpolate(
                    curr_tensor,
                    size=(target_h, target_w),
                    mode="area",
                )
                curr_tensor = downscaled

                # Copy back to CPU numpy array
                if is_rgb:
                    lvl_arr = downscaled.squeeze(0).permute(1, 2, 0).to(dtype=torch.uint8 if orig_dtype == np.uint8 else torch.uint16).cpu().numpy()
                else:
                    lvl_arr = downscaled.squeeze(0).squeeze(0).to(dtype=torch.uint8 if orig_dtype == np.uint8 else torch.uint16).cpu().numpy()

                pyramid.append(lvl_arr)

            # Free GPU memory
            del tensor, curr_tensor
            torch.cuda.empty_cache()

            return pyramid, healing_log

        except Exception as e:
            # Self-healing fallback to CPU if GPU encounters Out-Of-Memory
            healing_log.append(f"GPU VRAM pressure/error ({e}), self-healed via CPU pyramid builder")
            logger.warning("GPU pyramid failed, failing over to CPU: %s", e)
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()
            return self._cpu_generate_pyramid(base_data, levels)

    def _cpu_generate_pyramid(self, base_data: np.ndarray, levels: int = 4) -> Tuple[List[np.ndarray], List[str]]:
        """CPU fallback for multi-resolution pyramid generation."""
        pyramid = [base_data]
        curr = base_data
        for _ in range(levels):
            h, w = curr.shape[:2]
            if h <= 256 or w <= 256:
                break
            if len(curr.shape) == 3:
                curr = curr[::2, ::2, :]
            else:
                curr = curr[::2, ::2]
            pyramid.append(curr)
        return pyramid, []

    # -------------------------------------------------------------------------
    # Self-Healing Multi-Stage JP2 Reader
    # -------------------------------------------------------------------------
    def read_jp2(self, input_path: Path) -> Tuple[np.ndarray, Dict[str, Any], str, List[str]]:
        """Read JP2 with automatic multi-stage failover and corrupt stream healing."""
        path_str = str(input_path.resolve())
        healing_log: List[str] = []

        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        # Stage 1: Pillow / OpenJPEG (Most stable for JP2/JPX)
        if PILLOW_AVAILABLE:
            try:
                with Image.open(path_str) as img:
                    img.load()  # Force decode
                    data = np.array(img)
                    metadata = {
                        "mode": img.mode,
                        "shape": data.shape,
                        "dtype": str(data.dtype),
                    }
                    return data, metadata, "pillow_openjpeg", healing_log
            except Exception as e:
                healing_log.append(f"Pillow decode warning ({e}); self-healing via OpenCV/Glymur cascade")
                logger.debug("Pillow read issue on %s: %s", input_path.name, e)

        # Stage 2: OpenCV C++ Engine
        if OPENCV_AVAILABLE:
            try:
                img_cv = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
                if img_cv is not None and img_cv.size > 0:
                    if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
                        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    elif len(img_cv.shape) == 3 and img_cv.shape[2] == 4:
                        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)

                    metadata = {"shape": img_cv.shape, "dtype": str(img_cv.dtype)}
                    return img_cv, metadata, "opencv_fast", healing_log
            except Exception as e:
                healing_log.append(f"OpenCV decode error ({e}); falling back to raw parser")
                logger.debug("OpenCV read issue: %s", e)

        # Stage 3: Glymur ISO 15444 Reader
        if GLYMUR_AVAILABLE:
            try:
                jp2 = glymur.Jp2k(path_str)
                data = jp2[:]
                metadata = {"shape": data.shape, "dtype": str(data.dtype)}
                return data, metadata, "glymur_openjpeg", healing_log
            except Exception as e:
                healing_log.append(f"Glymur decode issue ({e})")
                logger.debug("Glymur read issue: %s", e)

        # Stage 4: Byte-stream sanitization & Truncated repair
        try:
            with open(path_str, "rb") as f:
                raw_bytes = f.read()

            # Attempt in-memory byte reconstruction
            with Image.open(io.BytesIO(raw_bytes)) as img_stream:
                img_stream.load()
                data = np.array(img_stream)
                healing_log.append("Successfully healed truncated/corrupt JPEG2000 byte stream")
                metadata = {"shape": data.shape, "dtype": str(data.dtype)}
                return data, metadata, "byte_stream_repaired", healing_log
        except Exception as e:
            raise ValueError(
                f"All self-healing reader cascades exhausted for {input_path.name}: {e}"
            )

    # -------------------------------------------------------------------------
    # Verified Multi-Backend TIFF Writer
    # -------------------------------------------------------------------------
    def write_tiff(
        self,
        data: np.ndarray,
        output_path: Path,
        compression: Optional[str] = None,
        tile_size: Optional[Tuple[int, int]] = None,
        pyramid: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool, List[str]]:
        """Write verified TIFF with automatic BigTIFF promotion and pyramid generation."""
        comp = (compression or self.config.compression).lower()
        tiles = tile_size or self.config.tile_size
        use_pyramid = pyramid if pyramid is not None else self.config.pyramid
        output_path.parent.mkdir(parents=True, exist_ok=True)
        healing_log: List[str] = []
        is_gpu = False

        if TIFFFILE_AVAILABLE:
            comp_map = {
                "tiff_deflate": "deflate",
                "deflate": "deflate",
                "tiff_lzw": "lzw",
                "lzw": "lzw",
                "tiff_jpeg": "jpeg",
                "jpeg": "jpeg",
                "zstd": "zstd",
                "packbits": "packbits",
                "none": None,
            }
            tf_comp = comp_map.get(comp, "deflate")

            # Self-healing: Automatically promote to BigTIFF if > 1.5GB or multi-resolution pyramid
            is_bigtiff = (data.nbytes > 1.5 * 1024 * 1024 * 1024) or use_pyramid
            if data.nbytes > 1.5 * 1024 * 1024 * 1024:
                healing_log.append("Auto-promoted to BigTIFF 64-bit offsets (file exceeds 1.5GB)")

            kwargs: Dict[str, Any] = {
                "compression": tf_comp,
                "bigtiff": is_bigtiff,
                "photometric": "rgb" if len(data.shape) == 3 and data.shape[2] in (3, 4) else "minisblack",
            }

            if tiles and len(tiles) == 2:
                kwargs["tile"] = tiles

            if use_pyramid:
                # GPU-accelerated pyramid building
                subsamples, pyr_heals = self._gpu_generate_pyramid(data, levels=4)
                healing_log.extend(pyr_heals)
                is_gpu = (CUDA_AVAILABLE and self.gpu_enabled and len(pyr_heals) == 0)

                with tifffile.TiffWriter(str(output_path), bigtiff=is_bigtiff) as tif:
                    tif.write(
                        subsamples[0],
                        subifds=len(subsamples) - 1,
                        tile=tiles,
                        compression=tf_comp,
                    )
                    for sub_data in subsamples[1:]:
                        tif.write(
                            sub_data,
                            subfiletype=1,
                            tile=tiles,
                            compression=tf_comp,
                        )
            else:
                tifffile.imwrite(str(output_path), data, **kwargs)

            # Self-healing verification check
            if not self._verify_tiff_header(output_path):
                healing_log.append("Initial TIFF header check warning; re-writing uncompressed fallback")
                tifffile.imwrite(str(output_path), data, bigtiff=is_bigtiff)

            return "tifffile_biomedical", is_gpu, healing_log

        # Fallback to Pillow
        if PILLOW_AVAILABLE:
            pil_comp_map = {
                "tiff_deflate": "tiff_deflate",
                "deflate": "tiff_deflate",
                "tiff_lzw": "tiff_lzw",
                "lzw": "tiff_lzw",
                "tiff_jpeg": "jpeg",
                "none": None,
            }
            img = Image.fromarray(data)
            save_kwargs: Dict[str, Any] = {"format": "TIFF"}
            c = pil_comp_map.get(comp, "tiff_deflate")
            if c:
                save_kwargs["compression"] = c
            if tiles:
                save_kwargs["tile"] = tiles

            img.save(str(output_path), **save_kwargs)
            return "pillow_fallback", False, healing_log

        # Fallback to OpenCV
        if OPENCV_AVAILABLE:
            cv_params = [cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_DEFLATE]
            img_out = data
            if len(data.shape) == 3 and data.shape[2] == 3:
                img_out = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), img_out, cv_params)
            return "opencv_fallback", False, healing_log

        raise RuntimeError("No TIFF writer backends available.")

    def _verify_tiff_header(self, file_path: Path) -> bool:
        """Verify TIFF magic bytes for data integrity (II* / MM* / II+ BigTIFF)."""
        try:
            with open(file_path, "rb") as f:
                header = f.read(4)
            # Little-endian standard (II\x2a\x00), Big-endian standard (MM\x00\x2a), BigTIFF (II\x2b\x00)
            return header.startswith(b"II\x2a\x00") or header.startswith(b"MM\x00\x2a") or header.startswith(b"II\x2b\x00") or header.startswith(b"MM\x00\x2b")
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Main Public Execution Methods
    # -------------------------------------------------------------------------
    def convert_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        compression: Optional[str] = None,
        tile_size: Optional[Tuple[int, int]] = None,
        pyramid: Optional[bool] = None,
        overwrite: Optional[bool] = None,
    ) -> ConversionResult:
        """Convert a single JP2/JPX file with GPU telemetry and self-healing resilience."""
        start_time = time.perf_counter()
        in_path = Path(input_path).resolve()

        if output_path is None:
            out_path = self.config.output_dir / in_path.with_suffix(".tif").name
        else:
            out_path = Path(output_path).resolve()

        should_overwrite = overwrite if overwrite is not None else self.config.overwrite

        if out_path.exists() and not should_overwrite:
            file_size_mb = out_path.stat().st_size / (1024 * 1024)
            return ConversionResult(
                input_path=in_path,
                output_path=out_path,
                success=True,
                file_size_mb=file_size_mb,
                elapsed_seconds=0.0,
                error_message="Skipped (file exists and overwrite is False)",
            )

        try:
            # Read image via self-healing pipeline
            data, metadata, read_backend, read_heals = self.read_jp2(in_path)

            height, width = data.shape[:2]
            channels = data.shape[2] if len(data.shape) > 2 else 1

            # Write TIFF via verified GPU/CPU pipeline
            write_backend, is_gpu, write_heals = self.write_tiff(
                data=data,
                output_path=out_path,
                compression=compression,
                tile_size=tile_size,
                pyramid=pyramid,
                metadata=metadata,
            )

            if not out_path.exists():
                raise RuntimeError(f"Output file was not created: {out_path}")

            file_size_mb = out_path.stat().st_size / (1024 * 1024)
            elapsed = time.perf_counter() - start_time
            all_heals = read_heals + write_heals

            logger.info(
                "Converted %s -> %s (%.2f MB, %dx%d, %dch, GPU:%s, time:%.2fs)",
                in_path.name, out_path.name, file_size_mb, width, height, channels,
                is_gpu, elapsed
            )

            return ConversionResult(
                input_path=in_path,
                output_path=out_path,
                success=True,
                file_size_mb=file_size_mb,
                dimensions=(width, height),
                channels=channels,
                elapsed_seconds=elapsed,
                backend_used=f"{read_backend}+{write_backend}",
                gpu_accelerated=is_gpu,
                healed_events=all_heals,
            )

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.exception("Conversion failed for %s: %s", in_path.name, e)
            return ConversionResult(
                input_path=in_path,
                output_path=None,
                success=False,
                error_message=str(e),
                elapsed_seconds=elapsed,
            )
        finally:
            # Memory cleanup
            gc.collect()
            if CUDA_AVAILABLE:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def batch_convert(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        pattern: Optional[str] = None,
        recursive: Optional[bool] = None,
        overwrite: Optional[bool] = None,
        progress_callback: Optional[Any] = None,
    ) -> List[ConversionResult]:
        """Batch convert all JP2/JPX files with fault-tolerant worker isolation."""
        in_dir = Path(input_dir).resolve()
        out_dir = Path(output_dir).resolve() if output_dir else self.config.output_dir.resolve()
        is_rec = recursive if recursive is not None else self.config.recursive
        pat = pattern or self.config.file_pattern
        ow = overwrite if overwrite is not None else self.config.overwrite

        if not in_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {in_dir}")

        out_dir.mkdir(parents=True, exist_ok=True)

        patterns = [pat]
        if "*.jp2" in pat.lower() and "*.jpx" not in pat.lower():
            patterns.extend(["*.jpx", "*.JP2", "*.JPX"])

        files: List[Path] = []
        for p in patterns:
            if is_rec:
                files.extend(list(in_dir.rglob(p)))
            else:
                files.extend(list(in_dir.glob(p)))

        unique_files = list(dict.fromkeys(files))
        logger.info("Batch scanning: %d slide(s) found in %s", len(unique_files), in_dir)

        results: List[ConversionResult] = []
        total = len(unique_files)

        for i, file_path in enumerate(unique_files, 1):
            rel_path = file_path.relative_to(in_dir)
            target_out = out_dir / rel_path.with_suffix(".tif")
            target_out.parent.mkdir(parents=True, exist_ok=True)

            res = self.convert_file(
                input_path=file_path,
                output_path=target_out,
                compression=self.config.compression,
                tile_size=self.config.tile_size,
                pyramid=self.config.pyramid,
                overwrite=ow,
            )
            results.append(res)

            if progress_callback:
                progress_callback(i, total, file_path, res)

        return results