"""Desktop GUI Application for JP2 to TIFF Converter Pro.

Modern Tkinter/TTK-based graphical interface with GPU acceleration,
self-healing auto-recovery, multi-threading, live diagnostics, and batch progress.
"""

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from jp2_tiff_converter.config import Config
from jp2_tiff_converter.converter import (
    CUDA_AVAILABLE,
    GLYMUR_AVAILABLE,
    GPU_DEVICE_NAME,
    OPENCV_AVAILABLE,
    PILLOW_AVAILABLE,
    TIFFFILE_AVAILABLE,
    ConversionResult,
    JP2Converter,
)
from jp2_tiff_converter.logging_config import setup_logging

logger = logging.getLogger("gui")

# Set Windows High DPI awareness if available
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


class GuiLogHandler(logging.Handler):
    """Custom logging handler to route logs to GUI text console."""

    def __init__(self, text_widget: ScrolledText):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        def append():
            try:
                self.text_widget.configure(state="normal")
                tag = "info"
                if record.levelno >= logging.ERROR:
                    tag = "error"
                elif record.levelno >= logging.WARNING or "self-healed" in msg.lower():
                    tag = "warning"
                elif "SUCCESS" in msg or "Converted" in msg:
                    tag = "success"

                self.text_widget.insert(tk.END, msg + "\n", tag)
                self.text_widget.see(tk.END)
                self.text_widget.configure(state="disabled")
            except Exception:
                pass

        self.text_widget.after(0, append)


class ConverterApp(tk.Tk):
    """Main Desktop Application Window."""

    def __init__(self, config: Optional[Config] = None):
        super().__init__()
        self.config = config or Config()
        self.converter = JP2Converter(self.config)

        self.title("OmniSlide Pro — GPU-Accelerated & Self-Healing Whole-Slide Suite")
        self.geometry("920x700")
        self.minsize(820, 600)

        # Style configuration
        self._setup_styles()

        # Build UI
        self._build_header()
        self._build_tabs()
        self._build_statusbar()

        # Wire logging
        self._setup_gui_logging()

    def _setup_styles(self) -> None:
        """Apply modern ttk styling."""
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.bg_color = "#f4f6f9"
        self.primary_color = "#1e40af"
        self.accent_color = "#0284c7"
        self.gpu_color = "#15803d"

        self.configure(bg=self.bg_color)
        self.style.configure(".", background=self.bg_color, font=("Segoe UI", 9))
        self.style.configure("TNotebook", background=self.bg_color)
        self.style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[16, 6])
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#1e293b", background=self.bg_color)
        self.style.configure("Subheader.TLabel", font=("Segoe UI", 9), foreground="#64748b", background=self.bg_color)

    def _build_header(self) -> None:
        """Create top banner header with GPU indicator badge."""
        header_frame = ttk.Frame(self, padding=(16, 12, 16, 8))
        header_frame.pack(fill="x")

        top_row = ttk.Frame(header_frame)
        top_row.pack(fill="x")

        title_label = ttk.Label(
            top_row,
            text="🔬 OmniSlide Pro",
            style="Header.TLabel",
        )
        title_label.pack(side="left")

        # GPU Badge
        if CUDA_AVAILABLE:
            gpu_badge_text = f"⚡ GPU Accelerated ({GPU_DEVICE_NAME})"
            badge_bg = "#dcfce7"
            badge_fg = "#166534"
        else:
            gpu_badge_text = "💻 CPU Mode"
            badge_bg = "#f1f5f9"
            badge_fg = "#475569"

        gpu_lbl = tk.Label(
            top_row,
            text=gpu_badge_text,
            bg=badge_bg,
            fg=badge_fg,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=3,
            relief="solid",
            bd=1,
        )
        gpu_lbl.pack(side="right")

        subtitle_label = ttk.Label(
            header_frame,
            text="Self-Healing Multi-Engine Pipeline for Whole-Slide Digital Pathology & Microscopy Scans",
            style="Subheader.TLabel",
        )
        subtitle_label.pack(anchor="w", pady=(2, 0))

    def _build_tabs(self) -> None:
        """Build tabbed navigation interface."""
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        # Tab 1: Single File
        self.tab_single = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(self.tab_single, text="  📄 Single File  ")
        self._build_single_tab()

        # Tab 2: Batch Directory
        self.tab_batch = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(self.tab_batch, text="  📁 Batch Directory  ")
        self._build_batch_tab()

        # Tab 3: Logs & Diagnostics
        self.tab_logs = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(self.tab_logs, text="  📋 Logs & Diagnostics  ")
        self._build_logs_tab()

        # Tab 4: Settings & Engines
        self.tab_settings = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(self.tab_settings, text="  ⚙️ Settings & Info  ")
        self._build_settings_tab()

    def _build_single_tab(self) -> None:
        """Single file conversion tab UI."""
        frame = self.tab_single

        # File selection box
        grp_files = ttk.LabelFrame(frame, text=" File Selection ", padding=12)
        grp_files.pack(fill="x", pady=6)

        ttk.Label(grp_files, text="Input JP2/JPX File:").grid(row=0, column=0, sticky="w", pady=4)
        self.single_input_var = tk.StringVar()
        ent_in = ttk.Entry(grp_files, textvariable=self.single_input_var, width=65)
        ent_in.grid(row=0, column=1, padx=6, pady=4, sticky="ew")
        btn_in = ttk.Button(grp_files, text="Browse...", command=self._browse_single_input)
        btn_in.grid(row=0, column=2, padx=4, pady=4)

        ttk.Label(grp_files, text="Output TIFF File:").grid(row=1, column=0, sticky="w", pady=4)
        self.single_output_var = tk.StringVar()
        ent_out = ttk.Entry(grp_files, textvariable=self.single_output_var, width=65)
        ent_out.grid(row=1, column=1, padx=6, pady=4, sticky="ew")
        btn_out = ttk.Button(grp_files, text="Browse...", command=self._browse_single_output)
        btn_out.grid(row=1, column=2, padx=4, pady=4)

        grp_files.columnconfigure(1, weight=1)

        # Options Box
        grp_opts = ttk.LabelFrame(frame, text=" Conversion Options ", padding=12)
        grp_opts.pack(fill="x", pady=6)

        ttk.Label(grp_opts, text="Compression:").grid(row=0, column=0, sticky="w", pady=4)
        self.single_comp_var = tk.StringVar(value=self.config.compression)
        comp_combo = ttk.Combobox(
            grp_opts,
            textvariable=self.single_comp_var,
            values=["tiff_deflate", "tiff_lzw", "tiff_jpeg", "zstd", "packbits", "none"],
            state="readonly",
            width=16,
        )
        comp_combo.grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(grp_opts, text="Tile Size:").grid(row=0, column=2, sticky="w", padx=(16, 4), pady=4)
        self.single_tile_var = tk.StringVar(value="256x256")
        tile_combo = ttk.Combobox(
            grp_opts,
            textvariable=self.single_tile_var,
            values=["256x256", "512x512", "1024x1024", "None"],
            state="readonly",
            width=12,
        )
        tile_combo.grid(row=0, column=3, sticky="w", padx=6, pady=4)

        self.single_pyramid_var = tk.BooleanVar(value=True)
        chk_pyramid = ttk.Checkbutton(grp_opts, text="⚡ Pyramidal BigTIFF (GPU-Accelerated Subsampling)", variable=self.single_pyramid_var)
        chk_pyramid.grid(row=1, column=0, columnspan=2, sticky="w", pady=6)

        self.single_overwrite_var = tk.BooleanVar(value=True)
        chk_ow = ttk.Checkbutton(grp_opts, text="Overwrite existing output file", variable=self.single_overwrite_var)
        chk_ow.grid(row=1, column=2, columnspan=2, sticky="w", pady=6)

        # Action Box
        btn_frame = ttk.Frame(frame, padding=8)
        btn_frame.pack(fill="x", pady=8)

        self.btn_convert_single = tk.Button(
            btn_frame,
            text="⚡ Convert to TIFF",
            font=("Segoe UI", 11, "bold"),
            bg=self.primary_color,
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            padx=20,
            pady=8,
            relief="flat",
            cursor="hand2",
            command=self._start_single_conversion,
        )
        self.btn_convert_single.pack(side="left", padx=4)

        self.btn_open_out_single = ttk.Button(
            btn_frame,
            text="📂 Open Output Folder",
            command=self._open_single_output_dir,
            state="disabled",
        )
        self.btn_open_out_single.pack(side="left", padx=8)

        # Single Result Info Frame
        self.single_info_frame = ttk.LabelFrame(frame, text=" Result & Telemetry ", padding=10)
        self.single_info_frame.pack(fill="both", expand=True, pady=6)

        self.lbl_single_result = ttk.Label(self.single_info_frame, text="Ready to convert.", font=("Segoe UI", 10))
        self.lbl_single_result.pack(anchor="w", pady=4)

    def _build_batch_tab(self) -> None:
        """Batch conversion tab UI."""
        frame = self.tab_batch

        grp_dirs = ttk.LabelFrame(frame, text=" Directories ", padding=12)
        grp_dirs.pack(fill="x", pady=6)

        ttk.Label(grp_dirs, text="Input Folder:").grid(row=0, column=0, sticky="w", pady=4)
        self.batch_input_var = tk.StringVar(value=str(self.config.input_dir))
        ent_bin = ttk.Entry(grp_dirs, textvariable=self.batch_input_var, width=65)
        ent_bin.grid(row=0, column=1, padx=6, pady=4, sticky="ew")
        btn_bin = ttk.Button(grp_dirs, text="Browse...", command=self._browse_batch_input)
        btn_bin.grid(row=0, column=2, padx=4, pady=4)

        ttk.Label(grp_dirs, text="Output Folder:").grid(row=1, column=0, sticky="w", pady=4)
        self.batch_output_var = tk.StringVar(value=str(self.config.output_dir))
        ent_bout = ttk.Entry(grp_dirs, textvariable=self.batch_output_var, width=65)
        ent_bout.grid(row=1, column=1, padx=6, pady=4, sticky="ew")
        btn_bout = ttk.Button(grp_dirs, text="Browse...", command=self._browse_batch_output)
        btn_bout.grid(row=1, column=2, padx=4, pady=4)

        grp_dirs.columnconfigure(1, weight=1)

        # Filter & Options
        grp_bopts = ttk.LabelFrame(frame, text=" Batch Options ", padding=10)
        grp_bopts.pack(fill="x", pady=6)

        ttk.Label(grp_bopts, text="Pattern:").grid(row=0, column=0, sticky="w", pady=4)
        self.batch_pattern_var = tk.StringVar(value="*.jp2")
        ent_pat = ttk.Entry(grp_bopts, textvariable=self.batch_pattern_var, width=12)
        ent_pat.grid(row=0, column=1, sticky="w", padx=6, pady=4)

        self.batch_rec_var = tk.BooleanVar(value=True)
        chk_brec = ttk.Checkbutton(grp_bopts, text="Search subfolders recursively", variable=self.batch_rec_var)
        chk_brec.grid(row=0, column=2, padx=12, sticky="w")

        self.batch_ow_var = tk.BooleanVar(value=False)
        chk_bow = ttk.Checkbutton(grp_bopts, text="Overwrite existing files", variable=self.batch_ow_var)
        chk_bow.grid(row=0, column=3, padx=12, sticky="w")

        # Progress bar
        self.batch_progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(frame, variable=self.batch_progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=(10, 4))

        self.lbl_batch_status = ttk.Label(frame, text="Ready for batch conversion.", font=("Segoe UI", 9, "italic"))
        self.lbl_batch_status.pack(anchor="w", pady=(0, 6))

        # Action Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=4)

        self.btn_batch_start = tk.Button(
            btn_frame,
            text="🚀 Start Batch Conversion",
            font=("Segoe UI", 11, "bold"),
            bg=self.primary_color,
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            padx=20,
            pady=8,
            relief="flat",
            cursor="hand2",
            command=self._start_batch_conversion,
        )
        self.btn_batch_start.pack(side="left", padx=4)

        self.btn_open_batch_out = ttk.Button(
            btn_frame,
            text="📂 Open Output Folder",
            command=self._open_batch_output_dir,
        )
        self.btn_open_batch_out.pack(side="left", padx=8)

        # Batch items list
        self.batch_tree = ttk.Treeview(
            frame,
            columns=("File", "Dimensions", "Size", "Duration", "Engine", "Status"),
            show="headings",
            height=6,
        )
        self.batch_tree.heading("File", text="File Name")
        self.batch_tree.heading("Dimensions", text="Dimensions")
        self.batch_tree.heading("Size", text="Output Size")
        self.batch_tree.heading("Duration", text="Time")
        self.batch_tree.heading("Engine", text="Engine & GPU")
        self.batch_tree.heading("Status", text="Status")

        self.batch_tree.column("File", width=200)
        self.batch_tree.column("Dimensions", width=110)
        self.batch_tree.column("Size", width=90)
        self.batch_tree.column("Duration", width=80)
        self.batch_tree.column("Engine", width=140)
        self.batch_tree.column("Status", width=110)

        self.batch_tree.pack(fill="both", expand=True, pady=6)

    def _build_logs_tab(self) -> None:
        """Logs and terminal console tab."""
        frame = self.tab_logs

        top_bar = ttk.Frame(frame)
        top_bar.pack(fill="x", pady=(0, 6))

        ttk.Label(top_bar, text="Application Execution & Self-Healing Telemetry Logs", font=("Segoe UI", 10, "bold")).pack(side="left")

        btn_clear = ttk.Button(top_bar, text="Clear Console", command=self._clear_logs)
        btn_clear.pack(side="right", padx=4)

        btn_diag = ttk.Button(top_bar, text="🔍 Run Diagnostics", command=self._run_diagnostics)
        btn_diag.pack(side="right", padx=4)

        self.log_text = ScrolledText(
            frame,
            wrap="word",
            bg="#0f172a",
            fg="#e2e8f0",
            insertbackground="#38bdf8",
            font=("Consolas", 9),
        )
        self.log_text.pack(fill="both", expand=True)

        self.log_text.tag_config("info", foreground="#94a3b8")
        self.log_text.tag_config("success", foreground="#4ade80", font=("Consolas", 9, "bold"))
        self.log_text.tag_config("warning", foreground="#fbbf24")
        self.log_text.tag_config("error", foreground="#f87171", font=("Consolas", 9, "bold"))

    def _build_settings_tab(self) -> None:
        """Settings and about information tab."""
        frame = self.tab_settings

        grp_eng = ttk.LabelFrame(frame, text=" Active Image Processing & Acceleration Backends ", padding=12)
        grp_eng.pack(fill="x", pady=6)

        engines = [
            ("NVIDIA CUDA GPU", CUDA_AVAILABLE, f"Hardware acceleration via {GPU_DEVICE_NAME}" if CUDA_AVAILABLE else "CUDA GPU not detected"),
            ("tifffile (Biomedical)", TIFFFILE_AVAILABLE, "Standard BigTIFF & Pyramidal WSI writer"),
            ("Pillow (OpenJPEG)", PILLOW_AVAILABLE, "ISO 15444 JPEG2000 reader with self-healing stream repair"),
            ("OpenCV (C++ accelerated)", OPENCV_AVAILABLE, "Vectorized high-speed computer vision engine"),
            ("Glymur Engine", GLYMUR_AVAILABLE, "Secondary JPEG2000 code-stream parser"),
        ]

        for i, (name, avail, desc) in enumerate(engines):
            status_text = "🟢 Active" if avail else "⚪ Not Installed"
            fg_col = "#16a34a" if avail else "#64748b"

            lbl_name = ttk.Label(grp_eng, text=f"{name}:", font=("Segoe UI", 9, "bold"))
            lbl_name.grid(row=i, column=0, sticky="w", pady=2, padx=4)

            lbl_st = tk.Label(grp_eng, text=status_text, fg=fg_col, font=("Segoe UI", 9, "bold"), bg=self.bg_color)
            lbl_st.grid(row=i, column=1, sticky="w", pady=2, padx=8)

            lbl_d = ttk.Label(grp_eng, text=desc, font=("Segoe UI", 9))
            lbl_d.grid(row=i, column=2, sticky="w", pady=2, padx=4)

        grp_info = ttk.LabelFrame(frame, text=" Self-Healing & Fault Tolerance Specifications ", padding=12)
        grp_info.pack(fill="x", pady=6)

        info_text = (
            "• Self-Healing Cascade: Automatic failover across 5 engine layers (CUDA -> OpenCV -> Pillow -> Glymur -> Raw).\n"
            "• Truncated Stream Repair: Recovers partial or non-standard JP2 files without crash.\n"
            "• BigTIFF Auto-Promotion: Seamlessly switches to 64-bit BigTIFF offsets when rasters exceed standard TIFF boundaries.\n"
            "• Out-of-Memory (OOM) Guard: Automatically frees VRAM cache and bounds memory allocations."
        )
        ttk.Label(grp_info, text=info_text, justify="left", font=("Segoe UI", 9)).pack(anchor="w")

        btn_save_cfg = ttk.Button(frame, text="💾 Save Current Settings to config.yaml", command=self._save_settings)
        btn_save_cfg.pack(anchor="w", pady=12)

    def _build_statusbar(self) -> None:
        """Bottom status indicator bar."""
        self.status_bar = ttk.Frame(self, padding=(16, 4))
        self.status_bar.pack(fill="x", side="bottom")

        self.lbl_status = ttk.Label(self.status_bar, text="Ready.", font=("Segoe UI", 8))
        self.lbl_status.pack(side="left")

        gpu_status_str = f"GPU: {GPU_DEVICE_NAME}" if CUDA_AVAILABLE else "GPU: Disabled (CPU mode)"
        ver_lbl = ttk.Label(self.status_bar, text=f"{gpu_status_str} | v2.0.0 Pro", font=("Segoe UI", 8), foreground="#64748b")
        ver_lbl.pack(side="right")

    def _setup_gui_logging(self) -> None:
        """Attach custom logging handler to console widget."""
        handler = GuiLogHandler(self.log_text)
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
        logger.info("JP2 to TIFF Converter Pro started (CUDA GPU: %s, Device: %s).", CUDA_AVAILABLE, GPU_DEVICE_NAME)

    # Single File Handlers
    def _browse_single_input(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select JPEG2000 File",
            filetypes=[("JPEG2000 Files", "*.jp2;*.jpx;*.JP2;*.JPX"), ("All Files", "*.*")],
        )
        if file_path:
            self.single_input_var.set(file_path)
            in_p = Path(file_path)
            out_p = in_p.with_suffix(".tif")
            self.single_output_var.set(str(out_p))

    def _browse_single_output(self) -> None:
        file_path = filedialog.asksaveasfilename(
            title="Select TIFF Output Path",
            defaultextension=".tif",
            filetypes=[("TIFF Files", "*.tif;*.tiff"), ("BigTIFF Files", "*.btf;*.tif"), ("All Files", "*.*")],
        )
        if file_path:
            self.single_output_var.set(file_path)

    def _start_single_conversion(self) -> None:
        in_str = self.single_input_var.get().strip()
        out_str = self.single_output_var.get().strip()

        if not in_str:
            messagebox.showwarning("Input Required", "Please choose an input JP2/JPX file.")
            return

        in_path = Path(in_str)
        if not in_path.exists():
            messagebox.showerror("File Not Found", f"Input file does not exist:\n{in_str}")
            return

        out_path = Path(out_str) if out_str else in_path.with_suffix(".tif")

        tile_str = self.single_tile_var.get()
        tile_size = None
        if "x" in tile_str:
            w, h = map(int, tile_str.split("x"))
            tile_size = (w, h)

        compression = self.single_comp_var.get()
        pyramid = self.single_pyramid_var.get()
        overwrite = self.single_overwrite_var.get()

        self.btn_convert_single.configure(state="disabled", text="⏳ Converting...")
        self.lbl_single_result.configure(text=f"Converting {in_path.name}...")
        self.lbl_status.configure(text=f"Converting {in_path.name}...")

        def worker():
            res = self.converter.convert_file(
                input_path=in_path,
                output_path=out_path,
                compression=compression,
                tile_size=tile_size,
                pyramid=pyramid,
                overwrite=overwrite,
            )

            def update_ui():
                self.btn_convert_single.configure(state="normal", text="⚡ Convert to TIFF")
                if res.success:
                    dims = f"{res.dimensions[0]}x{res.dimensions[1]}" if res.dimensions else "N/A"
                    gpu_txt = "⚡ CUDA GPU Accelerated" if res.gpu_accelerated else "💻 CPU Pipeline"
                    heals_txt = f"\n• Self-Healing Actions: {', '.join(res.healed_events)}" if res.healed_events else ""
                    info = (
                        f"✅ Conversion Succeeded!\n"
                        f"• Output File: {res.output_path}\n"
                        f"• Dimensions: {dims} (Channels: {res.channels})\n"
                        f"• Output Size: {res.file_size_mb:.2f} MB\n"
                        f"• Acceleration: {gpu_txt}\n"
                        f"• Time Elapsed: {res.elapsed_seconds:.2f}s (Engine: {res.backend_used}){heals_txt}"
                    )
                    self.lbl_single_result.configure(text=info)
                    self.lbl_status.configure(text=f"Converted {in_path.name} in {res.elapsed_seconds:.2f}s")
                    self.btn_open_out_single.configure(state="normal")
                    messagebox.showinfo("Success", f"Converted successfully!\nOutput: {out_path.name} ({res.file_size_mb:.2f} MB)")
                else:
                    self.lbl_single_result.configure(text=f"❌ Conversion Failed: {res.error_message}")
                    self.lbl_status.configure(text="Conversion failed.")
                    messagebox.showerror("Conversion Failed", f"Failed to convert file:\n{res.error_message}")

            self.after(0, update_ui)

        threading.Thread(target=worker, daemon=True).start()

    def _open_single_output_dir(self) -> None:
        out_str = self.single_output_var.get().strip()
        if out_str:
            p = Path(out_str)
            folder = p.parent if p.is_file() or not p.exists() else p
            if folder.exists():
                os.startfile(str(folder))

    # Batch Handlers
    def _browse_batch_input(self) -> None:
        folder = filedialog.askdirectory(title="Select Input Folder Containing JP2 Files")
        if folder:
            self.batch_input_var.set(folder)

    def _browse_batch_output(self) -> None:
        folder = filedialog.askdirectory(title="Select Output Folder For TIFF Files")
        if folder:
            self.batch_output_var.set(folder)

    def _start_batch_conversion(self) -> None:
        in_str = self.batch_input_var.get().strip()
        out_str = self.batch_output_var.get().strip()

        if not in_str or not Path(in_str).exists():
            messagebox.showerror("Error", "Please select a valid input directory.")
            return

        out_path = Path(out_str)
        out_path.mkdir(parents=True, exist_ok=True)

        pattern = self.batch_pattern_var.get().strip() or "*.jp2"
        recursive = self.batch_rec_var.get()
        overwrite = self.batch_ow_var.get()

        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)

        self.btn_batch_start.configure(state="disabled", text="⏳ Processing Batch...")
        self.batch_progress_var.set(0)
        self.lbl_batch_status.configure(text="Discovering files...")

        def on_progress(idx: int, total: int, file_path: Path, res: ConversionResult):
            pct = (idx / total) * 100
            def update():
                self.batch_progress_var.set(pct)
                status_txt = f"Processing [{idx}/{total}]: {file_path.name}"
                self.lbl_batch_status.configure(text=status_txt)
                self.lbl_status.configure(text=status_txt)

                dims = f"{res.dimensions[0]}x{res.dimensions[1]}" if res.dimensions else "-"
                size = f"{res.file_size_mb:.2f} MB" if res.success else "-"
                dur = f"{res.elapsed_seconds:.2f}s"
                eng = f"CUDA GPU" if res.gpu_accelerated else res.backend_used
                st = "SUCCESS" if res.success else "FAILED"

                self.batch_tree.insert("", "end", values=(file_path.name, dims, size, dur, eng, st))

            self.after(0, update)

        def worker():
            results = self.converter.batch_convert(
                input_dir=in_str,
                output_dir=out_path,
                pattern=pattern,
                recursive=recursive,
                overwrite=overwrite,
                progress_callback=on_progress,
            )

            def finish():
                self.btn_batch_start.configure(state="normal", text="🚀 Start Batch Conversion")
                succ = sum(1 for r in results if r.success)
                fail = len(results) - succ
                summary = f"Batch finished: {succ} succeeded, {fail} failed (Total: {len(results)})"
                self.lbl_batch_status.configure(text=summary)
                self.lbl_status.configure(text=summary)
                messagebox.showinfo("Batch Conversion Complete", summary)

            self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def _open_batch_output_dir(self) -> None:
        out_str = self.batch_output_var.get().strip()
        if out_str and Path(out_str).exists():
            os.startfile(out_str)

    def _clear_logs(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

    def _run_diagnostics(self) -> None:
        logger.info("--- SYSTEM DIAGNOSTICS & HARDWARE ACCELERATION AUDIT ---")
        logger.info("Python Version: %s", sys.version)
        logger.info("CUDA Hardware Acceleration: %s", "ENABLED (Active)" if CUDA_AVAILABLE else "DISABLED")
        logger.info("CUDA GPU Device: %s", GPU_DEVICE_NAME)
        logger.info("tifffile engine: %s", "AVAILABLE" if TIFFFILE_AVAILABLE else "UNAVAILABLE")
        logger.info("Pillow engine: %s", "AVAILABLE" if PILLOW_AVAILABLE else "UNAVAILABLE")
        logger.info("OpenCV engine: %s", "AVAILABLE" if OPENCV_AVAILABLE else "UNAVAILABLE")
        logger.info("Glymur engine: %s", "AVAILABLE" if GLYMUR_AVAILABLE else "UNAVAILABLE")
        logger.info("--- DIAGNOSTICS COMPLETE ---")

    def _save_settings(self) -> None:
        try:
            cfg_path = Path("config.yaml")
            self.config.compression = self.single_comp_var.get()
            self.config.pyramid = self.single_pyramid_var.get()
            self.config.to_file(cfg_path)
            messagebox.showinfo("Settings Saved", f"Settings successfully saved to {cfg_path.resolve()}")
            logger.info("Configuration saved to %s", cfg_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")


def launch_gui(config: Optional[Config] = None) -> None:
    """Launch the Desktop GUI application."""
    app = ConverterApp(config)
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
