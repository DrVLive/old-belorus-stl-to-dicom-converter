# raw_tuner.py — v2 (WL/CL + дефолты + перенос WL в DICOM)

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import numpy as np
from PIL import Image, ImageTk
import datetime, uuid

# ============ дефолты под твой кейс ============
DEFAULT_ROWS   = 1544
DEFAULT_COLS   = 1535
DEFAULT_BITS   = 16
DEFAULT_EXTRA  = 0
DEFAULT_ENDIAN = "LE"
DEFAULT_INVERT = True          # у тебя без инверта — «чёрный экран», значит исходник MONO1
DEFAULT_OFFSET = 10972
DEFAULT_AUTO_TAIL = False

# ============ утилиты ============

def to_uint8_window(arr16: np.ndarray, center: float, width: float) -> np.ndarray:
    """Окно/уровень → 8-бит для превью. center=Level, width=Window."""
    if width <= 1e-6:
        width = 1e-6
    lo = center - width / 2.0
    hi = center + width / 2.0
    a = np.clip(arr16.astype(np.float32), lo, hi)
    out = (a - lo) / (hi - lo)
    return (out * 255.0 + 0.5).astype(np.uint8)

def auto_wl_p1_p99(arr16: np.ndarray) -> tuple[float, float]:
    """Автоподбор окна по перцентилям p1–p99 (устойчиво к высветленным краям)."""
    p1, p99 = np.percentile(arr16, [1, 99]).astype(np.float32)
    width = max(1.0, float(p99 - p1))
    center = float((p1 + p99) / 2.0)
    return center, width

def score_no_banding(arr: np.ndarray) -> float:
    a = arr.astype(np.float32)
    dr = float(np.mean(np.abs(a[1:] - a[:-1]))) if a.shape[0] > 1 else 0.0
    dc = float(np.mean(np.abs(a[:, 1:] - a[:, :-1]))) if a.shape[1] > 1 else 0.0
    row_means = np.mean(a, axis=1) if a.size else np.array([0.0])
    band = float(np.var(row_means))
    vmin, vmax = float(a.min()), float(a.max())
    stick = float(np.mean(a <= vmin + 1) + np.mean(a >= vmax - 1))
    return dr * 0.75 + dc * 0.25 + band * 0.10 + stick * 10.0

# ============ запись SC DICOM ============
def write_sc_from_array(arr16: np.ndarray, out_path: Path,
                        photometric="MONOCHROME2",
                        wl_center: float | None = None,  # ### NEW
                        wl_width:  float | None = None):  # ### NEW
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    fm = FileMetaDataset()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.ImplementationClassUID = "1.2.826.0.1.3680043.10.543." + uuid.uuid4().hex[:12]

    ds = FileDataset(out_path.name, {}, file_meta=fm, preamble=b"\0"*128)
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.Modality = "OT"
    now = datetime.datetime.now()
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")

    ds.Rows, ds.Columns = int(arr16.shape[0]), int(arr16.shape[1])
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = arr16.astype(np.uint16).tobytes()

    # Window/Level — из GUI (если не задали — авто p1–p99)
    if wl_center is None or wl_width is None:
        wl_center, wl_width = auto_wl_p1_p99(arr16)
    ds.WindowCenter = [float(wl_center)]
    ds.WindowWidth  = [float(max(1.0, wl_width))]
    ds.VOILUTFunction = "LINEAR_EXACT"  # многие вьюеры уважают

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(out_path, write_like_original=False)

# ============ ядро реконструкции ============
def reconstruct(raw: memoryview,
               rows: int, cols: int, bits: int, extra: int,
               endian: str, offset_abs: int,
               transpose: bool, byte_swap: bool,
               invert_mono1: bool):
    try:
        bpp = 1 if bits == 8 else 2
        stride = cols * bpp + max(0, extra)
        need = rows * stride
        if offset_abs < 0 or offset_abs + need > len(raw):
            return False, None, "offset/need вне файла"

        view = raw[offset_abs:offset_abs + need]
        buf = bytearray(rows * cols * bpp)
        s = 0; d = 0
        for _ in range(rows):
            buf[d:d + cols * bpp] = view[s:s + cols * bpp]
            d += cols * bpp; s += stride

        if bits == 8:
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols).astype(np.uint16)
        else:
            dt = "<u2" if endian == "LE" else ">u2"
            arr = np.frombuffer(buf, dtype=dt).reshape(rows, cols)
            if byte_swap:
                arr = arr.byteswap().newbyteorder()
            arr = arr.astype(np.uint16)

        if transpose:
            arr = arr.T
        if invert_mono1:           # конвертируем MONO1 → MONOCHROME2
            arr = (arr.max() - arr).astype(np.uint16)
        return True, arr, ""
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"

# ============ GUI ============
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAW → preview (stride/offset/endian tuner)")
        self.geometry("1200x720")
        self.minsize(1000, 600)

        self.raw_path: Path|None = None
        self.raw_view: memoryview|None = None
        self.img_tk = None

        # параметры — стартуют твоими значениями
        self.rows = tk.IntVar(value=DEFAULT_ROWS)
        self.cols = tk.IntVar(value=DEFAULT_COLS)
        self.bits = tk.IntVar(value=DEFAULT_BITS)
        self.extra = tk.IntVar(value=DEFAULT_EXTRA)
        self.endian = tk.StringVar(value=DEFAULT_ENDIAN)
        self.transpose = tk.BooleanVar(value=False)
        self.byteswap = tk.BooleanVar(value=False)
        self.invert = tk.BooleanVar(value=DEFAULT_INVERT)
        self.offset = tk.IntVar(value=DEFAULT_OFFSET)     # абсолютный
        self.auto_to_tail = tk.BooleanVar(value=DEFAULT_AUTO_TAIL)

        # просмотр: zoom/pan
        self.zoom = tk.DoubleVar(value=100.0)      # %
        self.pan_x = tk.IntVar(value=0)            # px
        self.pan_y = tk.IntVar(value=0)            # px

        # Window/Level (яркость/контраст)  ### NEW
        self.auto_wl = tk.BooleanVar(value=True)
        self.wl_center = tk.DoubleVar(value=2048.0)
        self.wl_width  = tk.DoubleVar(value=4096.0)

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self); top.pack(side="top", fill="x", padx=8, pady=6)
        ttk.Button(top, text="Open file…", command=self.open_file).pack(side="left")
        ttk.Button(top, text="Save DICOM…", command=self.save_dicom).pack(side="left", padx=6)
        self.info_lbl = ttk.Label(top, text="—"); self.info_lbl.pack(side="left", padx=12)

        ctrl = ttk.Frame(self); ctrl.pack(side="left", fill="y", padx=8, pady=6)

        def add_row(lbl, widget):
            f = ttk.Frame(ctrl); f.pack(fill="x", pady=2)
            ttk.Label(f, text=lbl, width=14).pack(side="left")
            widget.pack(side="left", fill="x", expand=True)

        add_row("Rows",    ttk.Spinbox(ctrl, from_=64, to=4096, textvariable=self.rows, width=8, command=self.update_preview))
        add_row("Columns", ttk.Spinbox(ctrl, from_=64, to=4096, textvariable=self.cols,  width=8, command=self.update_preview))
        add_row("Bits",    ttk.Spinbox(ctrl, values=(8,16),      textvariable=self.bits, width=8, command=self.update_preview))
        add_row("Extra/row", ttk.Spinbox(ctrl, from_=0, to=128,  textvariable=self.extra, width=8, command=self.update_preview))

        f_end = ttk.Frame(ctrl); f_end.pack(fill="x", pady=2)
        ttk.Label(f_end, text="Endian", width=14).pack(side="left")
        ttk.Radiobutton(f_end, text="LE", variable=self.endian, value="LE", command=self.update_preview).pack(side="left")
        ttk.Radiobutton(f_end, text="BE", variable=self.endian, value="BE", command=self.update_preview).pack(side="left")

        f_chk = ttk.Frame(ctrl); f_chk.pack(fill="x", pady=2)
        ttk.Checkbutton(f_chk, text="Transpose",       variable=self.transpose, command=self.update_preview).pack(anchor="w")
        ttk.Checkbutton(f_chk, text="Byte-swap 16-bit", variable=self.byteswap,  command=self.update_preview).pack(anchor="w")
        ttk.Checkbutton(f_chk, text="Invert (MONO1→2)", variable=self.invert,    command=self.update_preview).pack(anchor="w")

        ttk.Separator(ctrl).pack(fill="x", pady=6)
        ttk.Checkbutton(ctrl, text="Auto offset = tail − need", variable=self.auto_to_tail, command=self.update_preview).pack(anchor="w")

        off_frame = ttk.Frame(ctrl); off_frame.pack(fill="x", pady=2)
        ttk.Label(off_frame, text="Offset (abs)", width=14).pack(side="left")
        e = ttk.Entry(off_frame, textvariable=self.offset, width=14); e.pack(side="left")
        e.bind("<Return>", lambda _ : self.update_preview())
        ttk.Button(off_frame, text="Snap offset to row", command=self.snap_offset_to_row).pack(side="left", padx=6)

        # View: zoom / pan
        ttk.Label(ctrl, text="View").pack(anchor="w", pady=(8,0))
        z = ttk.Scale(ctrl, from_=10, to=300, orient="horizontal", variable=self.zoom, command=lambda _ : self.update_preview())
        add_row("Zoom (%)", z)

        px = ttk.Scale(ctrl, from_=-2000, to=2000, orient="horizontal", variable=self.pan_x, command=lambda _ : self.update_preview())
        add_row("Pan X (px)", px)
        py = ttk.Scale(ctrl, from_=-2000, to=2000, orient="horizontal", variable=self.pan_y, command=lambda _ : self.update_preview())
        add_row("Pan Y (px)", py)
        ttk.Button(ctrl, text="Reset view", command=self.reset_view).pack(anchor="w", pady=2)

        # Window/Level (яркость/контраст)  ### NEW
        ttk.Separator(ctrl).pack(fill="x", pady=6)
        ttk.Label(ctrl, text="Window/Level").pack(anchor="w")
        ttk.Checkbutton(ctrl, text="Auto WL (p1→p99)", variable=self.auto_wl, command=self.update_preview).pack(anchor="w")
        wl_frame = ttk.Frame(ctrl); wl_frame.pack(fill="x", pady=2)
        ttk.Label(wl_frame, text="Level / Center", width=14).pack(side="left")
        ttk.Scale(wl_frame, from_=0, to=65535, orient="horizontal", variable=self.wl_center,
                  command=lambda _ : self.update_preview()).pack(side="left", fill="x", expand=True)
        ww_frame = ttk.Frame(ctrl); ww_frame.pack(fill="x", pady=2)
        ttk.Label(ww_frame, text="Width", width=14).pack(side="left")
        ttk.Scale(ww_frame, from_=1, to=65535, orient="horizontal", variable=self.wl_width,
                  command=lambda _ : self.update_preview()).pack(side="left", fill="x", expand=True)

        # метрика
        self.score_lbl = ttk.Label(ctrl, text="score: —")
        self.score_lbl.pack(anchor="w", pady=6)

        # canvas
        view = ttk.Frame(self); view.pack(side="right", fill="both", expand=True, padx=8, pady=6)
        self.canvas = tk.Canvas(view, bg="black")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _ : self.update_preview())

        # hotkeys
        self.bind("<Left>",  lambda _ : self.bump_offset(-4))
        self.bind("<Right>", lambda _ : self.bump_offset(+4))
        self.bind("<Up>",    lambda _ : self.bump_extra(+1))
        self.bind("<Down>",  lambda _ : self.bump_extra(-1))

    # --- helpers ---
    def reset_view(self):
        self.zoom.set(100.0); self.pan_x.set(0); self.pan_y.set(0)
        self.update_preview()

    def snap_offset_to_row(self):
        """Подровнять offset к началу строки (мод bpp)."""
        bpp = 1 if self.bits.get() == 8 else 2
        stride = self.cols.get() * bpp + max(0, self.extra.get())
        k = self.offset.get() % stride
        self.offset.set(self.offset.get() - k)
        self.update_preview()

    def open_file(self):
        p = filedialog.askopenfilename(title="Выберите файл (сырой/псевдо-DICOM/бинарник)")
        if not p: return
        b = Path(p).read_bytes()
        self.raw_path = Path(p)
        self.raw_view = memoryview(b)
        self.info_lbl.config(text=f"{self.raw_path.name}  |  {len(b):,} bytes".replace(",", " "))
        self.update_preview(initial=True)

    def bump_offset(self, delta: int):
        if self.auto_to_tail.get():
            self.auto_to_tail.set(False)
        self.offset.set(self.offset.get() + delta)
        self.update_preview()

    def bump_extra(self, delta: int):
        self.extra.set(max(0, self.extra.get() + delta))
        self.update_preview()

    def current_auto_offset(self) -> int:
        if not self.raw_view: return 0
        rows, cols, bits, extra = self.rows.get(), self.cols.get(), self.bits.get(), self.extra.get()
        bpp = 1 if bits == 8 else 2
        stride = cols * bpp + max(0, extra)
        need = rows * stride
        return max(0, len(self.raw_view) - need)

    def update_preview(self, initial: bool=False):
        if not self.raw_view:
            return
        if (self.auto_to_tail.get() or initial) and DEFAULT_AUTO_TAIL:
            self.offset.set(self.current_auto_offset())

        ok, arr16, msg = reconstruct(
            self.raw_view,
            rows=self.rows.get(), cols=self.cols.get(), bits=self.bits.get(),
            extra=self.extra.get(), endian=self.endian.get(),
            offset_abs=self.offset.get(),
            transpose=self.transpose.get(), byte_swap=self.byteswap.get(),
            invert_mono1=self.invert.get()
        )
        if not ok or arr16 is None or arr16.size == 0:
            self.score_lbl.config(text=f"error: {msg}")
            self.canvas.delete("all")
            return

        # WL (auto/manual)  ### NEW
        if self.auto_wl.get():
            c, w = auto_wl_p1_p99(arr16)
            self.wl_center.set(c); self.wl_width.set(max(1.0, w))
        c = float(self.wl_center.get()); w = float(max(1.0, self.wl_width.get()))
        im8 = to_uint8_window(arr16, c, w)

        sc = score_no_banding(arr16)
        self.score_lbl.config(text=f"score↓ {sc:.4f}   |   min={int(arr16.min())} max={int(arr16.max())}   |   off={self.offset.get()}")

        # масштаб с zoom/pan
        cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1: return
        h, w0 = im8.shape
        k = (self.zoom.get() / 100.0) * min(cw / w0, ch / h)
        new_w, new_h = max(1, int(w0 * k)), max(1, int(h * k))
        im_disp = Image.fromarray(im8).resize((new_w, new_h), Image.NEAREST)
        self.img_tk = ImageTk.PhotoImage(im_disp)
        self.canvas.delete("all")
        x = cw // 2 + int(self.pan_x.get())
        y = ch // 2 + int(self.pan_y.get())
        self.canvas.create_image(x, y, image=self.img_tk, anchor="center")

    def save_dicom(self):
        if not self.raw_view:
            messagebox.showwarning("Нет данных", "Сначала открой файл.")
            return
        ok, arr16, msg = reconstruct(
            self.raw_view,
            rows=self.rows.get(), cols=self.cols.get(), bits=self.bits.get(),
            extra=self.extra.get(), endian=self.endian.get(),
            offset_abs=self.offset.get(),
            transpose=self.transpose.get(), byte_swap=self.byteswap.get(),
            invert_mono1=self.invert.get()
        )
        if not ok or arr16 is None:
            messagebox.showerror("Ошибка", f"Невозможно собрать кадр: {msg}")
            return

        # WL из текущих ползунков  ### NEW
        center = float(self.wl_center.get())
        width  = float(max(1.0, self.wl_width.get()))

        p = filedialog.asksaveasfilename(
            title="Сохранить как DICOM",
            defaultextension=".dcm",
            filetypes=[("DICOM", "*.dcm"), ("All files", "*.*")]
        )
        if not p: return
        try:
            write_sc_from_array(arr16, Path(p), photometric="MONOCHROME2",
                                wl_center=center, wl_width=width)  # ### NEW
            messagebox.showinfo("Готово", f"Сохранено: {p}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

if __name__ == "__main__":
    App().mainloop()
