"""Tkinter front-end for the image vector converter."""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from PIL import Image, ImageDraw, ImageTk

from . import __version__
from .engine import Cancelled, ConverterSettings, convert, load_image, render
from .document import VectorDocument, save_eps, save_svg, to_eps

APP_TITLE = "Image Vector Converter"
IMAGE_TYPES = [
    ("All pictures", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff"),
    ("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("BMP", "*.bmp"),
    ("WebP", "*.webp"), ("TIFF", "*.tif *.tiff"), ("All files", "*.*"),
]

BG = "#24272e"
PANEL = "#2f333c"
FG = "#e6e8ec"
ACCENT = "#4c9ffe"

_WORKER_TICK_MS = 60
_AUTORUN_DELAY_MS = 550


class _Checker:
    """Chequerboard so transparent vector areas are obvious in the preview."""

    @staticmethod
    def build(width: int, height: int, square: int = 12) -> Image.Image:
        image = Image.new("RGB", (width, height), "#f4f4f4")
        draw = ImageDraw.Draw(image)
        for y in range(0, height, square):
            for x in range(0, width, square):
                if (x // square + y // square) % 2 == 0:
                    draw.rectangle([x, y, x + square - 1, y + square - 1], fill="#d8d8d8")
        return image


class VectorConverterApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP_TITLE} {__version__}")
        self.geometry("1200x780")
        self.minsize(1020, 660)
        self.configure(bg=BG)

        self.source_image: Optional[Image.Image] = None
        self.source_path: Optional[str] = None
        self.document: Optional[VectorDocument] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.messages: "queue.Queue[tuple]" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.cancel_flag = threading.Event()
        self.autorun_job: Optional[str] = None
        self.busy = False

        self._build_styles()
        self._build_menu()
        self._build_layout()
        self._bind_keys()
        self._set_status("Open a low quality image to begin (PNG / JPG / WEBP).")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ style
    def _build_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=PANEL, foreground=FG)
        style.configure("TFrame", background=PANEL)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=PANEL, foreground=FG)
        style.configure("Title.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Hint.TLabel", foreground="#a7adba", font=("Segoe UI", 8))
        style.configure("Stat.TLabel", foreground="#cfd4dd", font=("Consolas", 9))
        style.configure("TButton", padding=(10, 6))
        style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff")
        style.map("Accent.TButton",
                  background=[("active", "#3b86e6"), ("disabled", "#3a4658")])
        style.configure("TCheckbutton", background=PANEL, foreground=FG)
        style.configure("TRadiobutton", background=PANEL, foreground=FG)
        style.configure("TProgressbar", background=ACCENT, troughcolor="#20232a")
        style.configure("Vertical.TScale", background=PANEL)

    def _build_menu(self) -> None:
        menubar = tk.Menu(self, tearoff=0, bg=PANEL, fg=FG,
                          activebackground=ACCENT, activeforeground="#ffffff")
        file_menu = tk.Menu(menubar, tearoff=0, bg=PANEL, fg=FG,
                            activebackground=ACCENT, activeforeground="#ffffff")
        file_menu.add_command(label="Open image…", command=self.open_image,
                              accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Save as EPS (Illustrator)…",
                              command=lambda: self.save("eps"),
                              accelerator="Ctrl+S")
        file_menu.add_command(label="Save as SVG…",
                              command=lambda: self.save("svg"))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

        help_menu = tk.Menu(menubar, tearoff=0, bg=PANEL, fg=FG,
                            activebackground=ACCENT, activeforeground="#ffffff")
        help_menu.add_command(label="About", command=self._about)
        menubar.add_cascade(label="Help", menu=help_menu)

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x")
        ttk.Label(header, text="Image  →  Vector  (Illustrator EPS / SVG)",
                  style="Title.TLabel").pack(side="left")
        self.autoupdate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(header, text="Auto update preview", variable=self.autoupdate_var,
                        command=self._on_setting_changed).pack(side="right")

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True, pady=(8, 0))

        self._build_preview(body)
        self._build_controls(body)

        footer = ttk.Frame(root)
        footer.pack(fill="x", pady=(8, 0))
        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100, length=220)
        self.progress.pack(side="left")
        self.status_var = tk.StringVar(value="")
        ttk.Label(footer, textvariable=self.status_var, style="Hint.TLabel").pack(
            side="left", padx=(10, 0))
        self.stats_var = tk.StringVar(value="")
        ttk.Label(footer, textvariable=self.stats_var, style="Stat.TLabel").pack(side="right")

    def _build_preview(self, parent: ttk.Frame) -> None:
        left = ttk.Frame(parent)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        bar = ttk.Frame(left)
        bar.pack(fill="x", pady=(0, 6))
        self.view_var = tk.StringVar(value="vector")
        ttk.Radiobutton(bar, text="Vector result", value="vector", variable=self.view_var,
                        command=self._redraw).pack(side="left")
        ttk.Radiobutton(bar, text="Source image", value="source", variable=self.view_var,
                        command=self._redraw).pack(side="left", padx=(12, 0))
        ttk.Button(bar, text="Open image…", command=self.open_image).pack(side="right")

        self.canvas = tk.Canvas(left, bg="#1b1e24", highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())

    def _build_controls(self, parent: ttk.Frame) -> None:
        right = ttk.Frame(parent, width=330)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self.vars = {
            "colors": tk.DoubleVar(value=8),
            "tolerance": tk.DoubleVar(value=0.6),
            "corner_threshold": tk.DoubleVar(value=75.0),
            "min_area": tk.DoubleVar(value=6.0),
            "denoise": tk.DoubleVar(value=6),
            "upscale": tk.DoubleVar(value=1.5),
            "contrast": tk.DoubleVar(value=2),
            "sharpen": tk.DoubleVar(value=40),
            "despeckle": tk.DoubleVar(value=1),
            "drop_background": tk.BooleanVar(value=True),
            "background_tolerance": tk.DoubleVar(value=28.0),
        }

        vector_box = ttk.LabelFrame(right, text="Vector output", padding=8)
        vector_box.pack(fill="x", pady=(0, 8))
        self._slider(vector_box, "Colours", "colors", 2, 64, 1, "")
        self._slider(vector_box, "Detail accuracy", "tolerance", 0.2, 3.0, 0.05, "px",
                     hint="source pixels: low = more points, smoother curves")
        self._slider(vector_box, "Corner angle", "corner_threshold", 10, 120, 1, "°",
                     hint="lower = keeps sharp corners")
        self._slider(vector_box, "Remove specks", "min_area", 0, 400, 2, "px²",
                     hint="source px²: drops tiny dots and pinholes")

        repair_box = ttk.LabelFrame(right, text="Image repair (quality boost)", padding=8)
        repair_box.pack(fill="x", pady=(0, 8))
        self._slider(repair_box, "Denoise (JPEG)", "denoise", 0, 30, 1, "")
        self._slider(repair_box, "Upscale", "upscale", 1, 4, 0.5, "×",
                     hint="a minimum: small images are scaled up further")
        self._slider(repair_box, "Local contrast", "contrast", 0, 10, 1, "")
        self._slider(repair_box, "Sharpen", "sharpen", 0, 200, 5, "%")
        self._slider(repair_box, "Despeckle", "despeckle", 0, 10, 1, "")

        bg_box = ttk.LabelFrame(right, text="Background", padding=8)
        bg_box.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(bg_box, text="Remove background (transparent EPS)",
                        variable=self.vars["drop_background"],
                        command=self._on_setting_changed).pack(anchor="w")
        self._slider(bg_box, "Background tolerance", "background_tolerance", 4, 90, 1, "")

        actions = ttk.Frame(right)
        actions.pack(fill="x")
        self.convert_btn = ttk.Button(actions, text="Convert now",
                                      style="Accent.TButton", command=self.start_convert)
        self.convert_btn.pack(fill="x")
        row = ttk.Frame(actions)
        row.pack(fill="x", pady=(6, 0))
        self.eps_btn = ttk.Button(row, text="Save EPS", state="disabled",
                                  command=lambda: self.save("eps"))
        self.eps_btn.pack(side="left", fill="x", expand=True)
        self.svg_btn = ttk.Button(row, text="Save SVG", state="disabled",
                                  command=lambda: self.save("svg"))
        self.svg_btn.pack(side="left", fill="x", expand=True, padx=(6, 0))

    def _slider(self, parent: ttk.Frame, label: str, key: str, low: float, high: float,
                step: float, suffix: str, hint: str = "") -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        value_label = ttk.Label(row, width=8, anchor="e", style="Stat.TLabel")
        value_label.pack(side="right")
        ttk.Label(row, text=label, width=19, anchor="w").pack(side="left")

        var = self.vars[key]
        scale = ttk.Scale(row, from_=low, to=high, variable=var,
                          orient="horizontal", command=lambda _v: self._on_scale(key))
        scale.pack(side="left", fill="x", expand=True, padx=(6, 6))

        def refresh(*_args):
            value = var.get()
            value_label.configure(text=f"{value:g}{suffix}")

        var.trace_add("write", refresh)
        refresh()

        if hint:
            ttk.Label(parent, text=hint, style="Hint.TLabel").pack(anchor="w", pady=(0, 4))

    # ------------------------------------------------------------------ events
    def _bind_keys(self) -> None:
        self.bind("<Control-o>", lambda _e: self.open_image())
        self.bind("<Control-O>", lambda _e: self.open_image())
        self.bind("<Control-s>", lambda _e: self.save("eps"))
        self.bind("<Control-S>", lambda _e: self.save("eps"))

    def _on_scale(self, _key: str) -> None:
        self._on_setting_changed()

    def _on_setting_changed(self) -> None:
        if self.source_image is None:
            return
        if self.autorun_job:
            try:
                self.after_cancel(self.autorun_job)
            except tk.TclError:
                pass
            self.autorun_job = None
        if self.autoupdate_var.get():
            self.autorun_job = self.after(_AUTORUN_DELAY_MS, self.start_convert)

    def settings(self) -> ConverterSettings:
        v = self.vars
        return ConverterSettings(
            colors=int(round(v["colors"].get())),
            tolerance=float(v["tolerance"].get()),
            corner_threshold=float(v["corner_threshold"].get()),
            min_area=float(v["min_area"].get()),
            smoothing=2,
            drop_background=bool(v["drop_background"].get()),
            background_tolerance=float(v["background_tolerance"].get()),
            denoise=int(round(v["denoise"].get())),
            upscale=float(v["upscale"].get()),
            contrast=int(round(v["contrast"].get())),
            sharpen=int(round(v["sharpen"].get())),
            despeckle=int(round(v["despeckle"].get())),
        )

    # ------------------------------------------------------------------ actions
    def open_image(self) -> None:
        path = filedialog.askopenfilename(title="Open image", filetypes=IMAGE_TYPES)
        if not path:
            return
        try:
            image = load_image(path)
        except Exception as exc:  # noqa: BLE001 - report to the user
            messagebox.showerror("Cannot open image", str(exc))
            return
        self.source_image = image
        self.source_path = path
        self.document = None
        self.title(f"{APP_TITLE} - {os.path.basename(path)}")
        self.eps_btn.configure(state="disabled")
        self.svg_btn.configure(state="disabled")
        self.stats_var.set("")
        self.view_var.set("source")
        self._redraw()
        self._set_status(f"Loaded {image.size[0]}×{image.size[1]} px. "
                         "Adjust the sliders, then Convert.")
        self.start_convert()

    def start_convert(self) -> None:
        if self.source_image is None or self.busy:
            if self.busy:
                self.cancel_flag.set()
            return
        self.busy = True
        self.cancel_flag.clear()
        self.convert_btn.configure(text="Cancel")
        settings = self.settings()
        image = self.source_image.copy()
        name = os.path.basename(self.source_path or "image")
        self.messages = queue.Queue()

        def report(percent: int, message: str) -> None:
            self.messages.put(("progress", percent, message))

        def worker() -> None:
            try:
                document = convert(image, settings, report,
                                   cancel=self.cancel_flag.is_set, source_name=name)
                eps_size = len(to_eps(document).encode("ascii", "ignore"))
                self.messages.put(("done", document, eps_size))
            except Cancelled:
                self.messages.put(("cancelled",))
            except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                self.messages.put(("error", exc))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()
        self.after(_WORKER_TICK_MS, self._poll)

    def _poll(self) -> None:
        try:
            while True:
                item = self.messages.get_nowait()
                kind = item[0]
                if kind == "progress":
                    self.progress.configure(value=item[1])
                    self._set_status(item[2])
                elif kind == "done":
                    self._finish(item[1], item[2])
                    return
                elif kind == "cancelled":
                    self.busy = False
                    self.worker = None
                    self.convert_btn.configure(text="Convert now")
                    self._set_status("Cancelled.")
                    return
                elif kind == "error":
                    self._finish_error(item[1])
                    return
        except queue.Empty:
            pass
        if self.busy:
            self.after(_WORKER_TICK_MS, self._poll)

    def _finish(self, document: VectorDocument, eps_size: int) -> None:
        self.busy = False
        self.worker = None
        self.document = document
        self.progress.configure(value=100)
        self.convert_btn.configure(text="Convert now")
        self.eps_btn.configure(state="normal")
        self.svg_btn.configure(state="normal")
        self._set_status(f"Vector ready: {document.width}×{document.height} px, "
                         f"EPS ≈ {eps_size / 1024:.0f} KB")
        self.stats_var.set(
            f"colours {len(document.layers)}   paths {document.path_count}   "
            f"anchors {document.anchor_count}   scale {document.trace_scale:g}×")
        if self.view_var.get() == "source":
            self.view_var.set("vector")
        self._redraw()

    def _finish_error(self, exc: Exception) -> None:
        self.busy = False
        self.worker = None
        self.convert_btn.configure(text="Convert now")
        self._set_status("Conversion failed.")
        messagebox.showerror("Conversion failed", str(exc))

    def save(self, kind: str) -> None:
        if self.document is None:
            messagebox.showinfo(APP_TITLE, "Convert an image first.")
            return
        base = os.path.splitext(os.path.basename(self.source_path or "image"))[0]
        if kind == "eps":
            path = filedialog.asksaveasfilename(
                title="Save EPS", defaultextension=".eps", initialfile=f"{base}.eps",
                filetypes=[("EPS for Illustrator", "*.eps"), ("All files", "*.*")])
        else:
            path = filedialog.asksaveasfilename(
                title="Save SVG", defaultextension=".svg", initialfile=f"{base}.svg",
                filetypes=[("SVG", "*.svg"), ("All files", "*.*")])
        if not path:
            return
        try:
            if kind == "eps":
                save_eps(self.document, path)
            else:
                save_svg(self.document, path)
        except OSError as exc:
            messagebox.showerror("Cannot save", str(exc))
            return
        size_kb = os.path.getsize(path) / 1024.0
        self._set_status(f"Saved {os.path.basename(path)} ({size_kb:.0f} KB).")
        if self.source_path:
            try:
                ratio = 100.0 * (1.0 - size_kb / max(os.path.getsize(self.source_path) / 1024.0, 1e-6))
                self._set_status(f"Saved {os.path.basename(path)} - {size_kb:.0f} KB "
                                 f"(~{ratio:.0f}% smaller than the bitmap)")
            except OSError:
                pass

    # ------------------------------------------------------------------ preview
    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self.source_image is None:
            self.canvas.create_text(
                self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
                text="Open a low quality image\n(PNG / JPG / WEBP / BMP)",
                fill="#7d8492", font=("Segoe UI", 12), justify="center")
            return

        if self.view_var.get() == "source" or self.document is None:
            image = self.source_image.convert("RGB")
        else:
            image = render(self.document, max_size=(1000, 1000),
                           background=None if self.document.is_transparent else (255, 255, 255))

        width, height = image.size
        canvas_w = max(1, self.canvas.winfo_width() - 16)
        canvas_h = max(1, self.canvas.winfo_height() - 16)
        scale = min(canvas_w / width, canvas_h / height, 1.0)
        target = (max(1, int(width * scale)), max(1, int(height * scale)))
        shown = image.resize(target, Image.LANCZOS) if target != (width, height) else image

        if shown.mode == "RGBA":
            background = _Checker.build(*target)
            background.paste(shown, (0, 0), shown)
            shown = background

        self.preview_photo = ImageTk.PhotoImage(shown)
        self.canvas.create_image(self.canvas.winfo_width() / 2,
                                 self.canvas.winfo_height() / 2,
                                 image=self.preview_photo)

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _about(self) -> None:
        messagebox.showinfo(
            f"About {APP_TITLE}",
            f"{APP_TITLE} {__version__}\n\n"
            "Repairs a low quality bitmap (denoise, despeckle, upscale, sharpen),\n"
            "reduces it to flat colours, then traces every colour region into\n"
            "smooth cubic Bezier paths and writes an Illustrator-ready EPS\n"
            "(or SVG) file.\n\n"
            "Output is true vector geometry: infinitely scalable, editable paths.")

    def _on_close(self) -> None:
        self.cancel_flag.set()
        self.destroy()


def main() -> None:
    app = VectorConverterApp()
    app.mainloop()
