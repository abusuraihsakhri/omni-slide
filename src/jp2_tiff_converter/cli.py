"""Command-line interface for JP2 to TIFF converter."""

import sys
from pathlib import Path
from typing import Optional, Tuple

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from jp2_tiff_converter.config import DEFAULT_CONFIG_YAML, Config
from jp2_tiff_converter.converter import (
    CUDA_AVAILABLE,
    GLYMUR_AVAILABLE,
    GPU_DEVICE_NAME,
    OPENCV_AVAILABLE,
    PILLOW_AVAILABLE,
    TIFFFILE_AVAILABLE,
    JP2Converter,
)
from jp2_tiff_converter.logging_config import get_logger, setup_logging

console = Console()
logger = get_logger("cli")


@click.group()
@click.version_option(version="2.0.0", prog_name="jp2-tiff-converter")
@click.option("--config", "-c", type=click.Path(path_type=Path), help="Path to config file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose debug logging")
@click.pass_context
def cli(ctx: click.Context, config: Optional[Path], verbose: bool) -> None:
    """JP2 to TIFF Converter Pro — GPU-Accelerated & Self-Healing Microscopy Suite."""
    ctx.ensure_object(dict)

    if config and config.exists():
        cfg = Config.from_file(config)
    else:
        cfg = Config()

    if verbose:
        cfg.log_level = "DEBUG"

    setup_logging(cfg)
    ctx.obj["config"] = cfg


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path), required=False)
@click.option(
    "--compression",
    type=click.Choice(["tiff_deflate", "tiff_lzw", "tiff_jpeg", "zstd", "packbits", "none"]),
    help="TIFF compression algorithm",
)
@click.option("--tile-size", type=(int, int), default=(256, 256), help="Tile width and height")
@click.option("--pyramid/--no-pyramid", default=True, help="Generate multi-resolution pyramidal BigTIFF")
@click.option("--overwrite/--no-overwrite", default=True, help="Overwrite output file if exists")
@click.pass_context
def convert(
    ctx: click.Context,
    input_path: Path,
    output_path: Optional[Path],
    compression: Optional[str],
    tile_size: Tuple[int, int],
    pyramid: bool,
    overwrite: bool,
) -> None:
    """Convert a single JP2/JPX file to TIFF format."""
    cfg = ctx.obj["config"]

    if compression:
        cfg.compression = compression
    cfg.tile_size = tile_size
    cfg.pyramid = pyramid

    if output_path is None:
        output_path = cfg.output_dir / input_path.with_suffix(".tif").name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter = JP2Converter(cfg)

    gpu_str = f"[green]⚡ GPU ({GPU_DEVICE_NAME})[/green]" if CUDA_AVAILABLE else "[cyan]💻 CPU[/cyan]"
    with console.status(f"Converting {input_path.name} using {gpu_str}..."):
        result = converter.convert_file(
            input_path=input_path,
            output_path=output_path,
            compression=cfg.compression,
            tile_size=tile_size,
            pyramid=pyramid,
            overwrite=overwrite,
        )

    if result.success:
        dims_str = f"{result.dimensions[0]}x{result.dimensions[1]}" if result.dimensions else "N/A"
        accel_str = "[green]⚡ CUDA GPU[/green]" if result.gpu_accelerated else "[cyan]CPU[/cyan]"
        heals_str = f" ([yellow]Healed: {len(result.healed_events)}[/yellow])" if result.healed_events else ""
        console.print(
            f"[green]✓ Success:[/green] {output_path.name} "
            f"([cyan]{result.file_size_mb:.2f} MB[/cyan], [yellow]{dims_str}[/yellow], "
            f"{accel_str} {result.backend_used} in [bold green]{result.elapsed_seconds:.2f}s[/bold green]{heals_str})"
        )
    else:
        console.print(f"[red]✗ Failed:[/red] {result.error_message}")
        sys.exit(1)


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=False)
@click.argument("output_dir", type=click.Path(path_type=Path), required=False)
@click.option("--pattern", "-p", default="*.jp2", help="File pattern matching (e.g. *.jp2)")
@click.option("--recursive/--no-recursive", default=True, help="Search directories recursively")
@click.option("--compression", help="TIFF compression format")
@click.option("--pyramid/--no-pyramid", default=True, help="Generate multi-resolution pyramid")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files")
@click.pass_context
def batch(
    ctx: click.Context,
    input_dir: Optional[Path],
    output_dir: Optional[Path],
    pattern: str,
    recursive: bool,
    compression: Optional[str],
    pyramid: bool,
    overwrite: bool,
) -> None:
    """Batch convert all JP2/JPX files in a directory."""
    cfg = ctx.obj["config"]

    if input_dir:
        cfg.input_dir = input_dir
    if output_dir:
        cfg.output_dir = output_dir
    if compression:
        cfg.compression = compression
    cfg.file_pattern = pattern
    cfg.recursive = recursive
    cfg.pyramid = pyramid
    cfg.overwrite = overwrite

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    converter = JP2Converter(cfg)

    files = []
    patterns = [pattern]
    if "*.jp2" in pattern.lower():
        patterns.extend(["*.jpx", "*.JP2", "*.JPX"])

    for p in patterns:
        if recursive:
            files.extend(list(cfg.input_dir.rglob(p)))
        else:
            files.extend(list(cfg.input_dir.glob(p)))

    files = list(dict.fromkeys(files))

    if not files:
        console.print(f"[yellow]No files matching '{pattern}' found in {cfg.input_dir}[/yellow]")
        return

    gpu_str = f"[green]⚡ GPU ({GPU_DEVICE_NAME})[/green]" if CUDA_AVAILABLE else "[cyan]💻 CPU[/cyan]"
    console.print(f"Discovered [cyan]{len(files)}[/cyan] slide(s). Engine: {gpu_str}")

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting...", total=len(files))

        for file_path in files:
            progress.update(task, description=f"Processing {file_path.name}")
            rel_path = file_path.relative_to(cfg.input_dir)
            out_file = cfg.output_dir / rel_path.with_suffix(".tif")
            out_file.parent.mkdir(parents=True, exist_ok=True)

            res = converter.convert_file(
                input_path=file_path,
                output_path=out_file,
                compression=cfg.compression,
                tile_size=cfg.tile_size,
                pyramid=cfg.pyramid,
                overwrite=overwrite,
            )
            results.append(res)
            progress.advance(task)

    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    table = Table(title="Batch Conversion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Total Files Processed", str(len(results)))
    table.add_row("Successful", f"[green]{successful}[/green]")
    table.add_row("Failed", f"[red]{failed}[/red]")
    table.add_row("Hardware Engine", f"[green]NVIDIA CUDA GPU ({GPU_DEVICE_NAME})[/green]" if CUDA_AVAILABLE else "CPU Multi-threaded")
    table.add_row("Output Directory", str(cfg.output_dir))

    console.print(table)

    if failed > 0:
        console.print("\n[red]Failed Files List:[/red]")
        for r in results:
            if not r.success:
                console.print(f"  ❌ {r.input_path.name}: {r.error_message}")
        sys.exit(1)


@cli.command()
@click.pass_context
def gui(ctx: click.Context) -> None:
    """Launch the Desktop Graphical User Interface (GUI)."""
    from jp2_tiff_converter.gui import launch_gui
    cfg = ctx.obj["config"]
    console.print("[green]Launching Desktop Application GUI...[/green]")
    launch_gui(cfg)


@cli.command()
@click.argument("config_path", type=click.Path(path_type=Path), required=False)
@click.pass_context
def init_config(ctx: click.Context, config_path: Optional[Path]) -> None:
    """Create a default configuration file."""
    if config_path is None:
        config_path = Path("config.yaml")

    if config_path.exists() and not click.confirm(f"{config_path} exists. Overwrite?"):
        return

    Config.create_default(config_path)
    console.print(f"[green]Created default configuration at {config_path}[/green]")
    console.print(DEFAULT_CONFIG_YAML)


@cli.command()
@click.pass_context
def check_deps(ctx: click.Context) -> None:
    """Audit and display available image conversion backends & GPU status."""
    table = Table(title="Available Image Processing & Hardware Acceleration Engines")
    table.add_column("Backend / Hardware", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Capabilities & Notes")

    table.add_row(
        "NVIDIA CUDA GPU",
        f"[green]Active ({GPU_DEVICE_NAME})[/green]" if CUDA_AVAILABLE else "[yellow]CPU Only[/yellow]",
        "Ultra-fast tensor subsampling, color conversions & pyramid builds" if CUDA_AVAILABLE else "CUDA not available",
    )
    table.add_row(
        "tifffile",
        "[green]Available (Active)[/green]" if TIFFFILE_AVAILABLE else "[red]Not installed[/red]",
        "Primary biomedical BigTIFF, tiled TIFF, pyramid writer",
    )
    table.add_row(
        "Pillow (PIL)",
        "[green]Available (Active)[/green]" if PILLOW_AVAILABLE else "[red]Not installed[/red]",
        "Universal JPEG2000 reader with self-healing stream recovery",
    )
    table.add_row(
        "OpenCV (cv2)",
        "[green]Available (Active)[/green]" if OPENCV_AVAILABLE else "[red]Not installed[/red]",
        "High-performance C++ accelerated computer vision backend",
    )
    table.add_row(
        "glymur",
        "[green]Available[/green]" if GLYMUR_AVAILABLE else "[yellow]OpenJPEG library needed[/yellow]",
        "ISO 15444 JPEG2000 code-stream parser",
    )

    console.print(table)


if __name__ == "__main__":
    cli()