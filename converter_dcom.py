# batch_raw_to_dicom.py
# Пакетный GUI-конвертер "сырых" кадров/псевдо-DICOM → валидный DICOM (SC).
# Сохраняет .dcm рядом с исходником (то же имя, другое расширение).

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading, uuid, datetime
import numpy as np

# ---------- ДЕФОЛТЫ (как на скрине) ----------
DEFAULT_ROWS   = 1450
DEFAULT_COLS   = 1535
DEFAULT_BITS   = 16
DEFAULT_EXTRA  = 0
DEFAULT_ENDIAN = "BE"        # Big Endian по умолчанию
DEFAULT_INVERT = True
DEFAULT_AUTO_OFFSET = False  # автосмещение выкл.
DEFAULT_OFFSET = 10972

# WL (ручные; «Auto WL» по умолчанию выкл.)
DEFAULT_WL_CENTER = 26000.0
DEFAULT_WL_WIDTH  = 18000.0

# ---------- утилиты ----------
def auto_wl_p1_p99(arr16: np.ndarray) -> tuple[float, float]:
    p1, p99 = np.percentile(arr16, [1, 99]).astype(np.float32)
    w = float(max(1.0, p99 - p1))
    c = float((p1 + p99) / 2.0)
    return c, w

def write_sc_from_array(arr16: np.ndarray, out_path: Path,
                        photometric="MONOCHROME2",
                        wl_center: float | None = None,
                        wl_width: float | None = None):
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

    if wl_center is None or wl_width is None:
        wl_center, wl_width = auto_wl_p1_p99(arr16)
    ds.WindowCenter = [float(wl_center)]
    ds.WindowWidth  = [float(max(1.0, wl_width))]
    ds.VOILUTFunction = "LINEAR_EXACT"

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(out_path, write_like_original=False)

def reconstruct_from_raw(raw: memoryview,
                         rows: int, cols: int, bits: int, extra: int,
                         endian: str, offset_abs: int,
                         invert_mono1: bool) -> np.ndarray:
    bpp = 1 if bits == 8 else 2
    stride = cols * bpp + max(0, extra)
    need = rows * stride
    if offset_abs < 0 or offset_abs + need > len(raw):
        raise ValueError("offset/need вне файла")
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
        arr = np.frombuffer(buf, dtype=dt).reshape(rows, cols).astype(np.uint16)

    if invert_mono1:
        arr = (arr.max() - arr).astype(np.uint16)
    return arr

def auto_offset(len_bytes: int, rows: int, cols: int, bits: int, extra: int) -> int:
    bpp = 1 if bits == 8 else 2
    stride = cols * bpp + max(0, extra)
    need = rows * stride
    return max(0, len_bytes - need)

# ---------- простой тултип ----------
class Tooltip:
    def __init__(self, widget, text: str, delay_ms: int = 400):
        self.widget = widget
        self.text = text
        self.tip = None
        self.after_id = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)

    def _schedule(self, _):
        self.after_id = self.widget.after(400, self._show)

    def _show(self):
        if self.tip: return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(self.tip, text=self.text, justify="left",
                       background="#FFFFE0", relief="solid", borderwidth=1,
                       font=("TkDefaultFont", 9))
        lbl.pack(ipadx=6, ipady=3)

    def _hide(self, _):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.tip:
            self.tip.destroy()
            self.tip = None

# ---------- GUI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAW → DICOM (batch)")
        self.geometry("820x560")
        self.minsize(820, 560)

        self.src_path = tk.StringVar()
        self.in_rows   = tk.IntVar(value=DEFAULT_ROWS)
        self.in_cols   = tk.IntVar(value=DEFAULT_COLS)
        self.in_bits   = tk.IntVar(value=DEFAULT_BITS)
        self.in_extra  = tk.IntVar(value=DEFAULT_EXTRA)
        self.in_endian = tk.StringVar(value=DEFAULT_ENDIAN)
        self.in_invert = tk.BooleanVar(value=DEFAULT_INVERT)

        self.auto_off  = tk.BooleanVar(value=DEFAULT_AUTO_OFFSET)
        self.in_offset = tk.IntVar(value=DEFAULT_OFFSET)

        self.auto_wl   = tk.BooleanVar(value=False)  # Auto WL OFF by default
        self.wl_center = tk.DoubleVar(value=DEFAULT_WL_CENTER)
        self.wl_width  = tk.DoubleVar(value=DEFAULT_WL_WIDTH)

        self._build_ui()
        self._worker = None

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # Источник
        frm_in = ttk.LabelFrame(self, text="Источник")
        frm_in.pack(fill="x", **pad)
        ttk.Entry(frm_in, textvariable=self.src_path).grid(row=0, column=0, columnspan=4, sticky="ew", padx=4)
        frm_in.columnconfigure(0, weight=1)
        ttk.Button(frm_in, text="Выбрать файл…", command=self.pick_file).grid(row=0, column=4, padx=2)
        ttk.Button(frm_in, text="Выбрать папку…", command=self.pick_folder).grid(row=0, column=5)

        # Параметры входа
        frm_p = ttk.LabelFrame(self, text="Параметры кадра")
        frm_p.pack(fill="x", **pad)

        ttk.Label(frm_p, text="Rows").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(frm_p, from_=64, to=4096, textvariable=self.in_rows, width=8).grid(row=0, column=1, padx=4)

        lbl_cols = ttk.Label(frm_p, text="Columns")
        lbl_cols.grid(row=0, column=2, sticky="w")
        sp_cols = ttk.Spinbox(frm_p, from_=64, to=4096, textvariable=self.in_cols, width=8)
        sp_cols.grid(row=0, column=3, padx=4)
        Tooltip(sp_cols, "менять только после просмотра в визере конвертера")

        ttk.Label(frm_p, text="Bits").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(frm_p, values=(8,16), textvariable=self.in_bits, width=8).grid(row=0, column=5, padx=4)

        ttk.Label(frm_p, text="Extra/row").grid(row=0, column=6, sticky="w")
        ttk.Spinbox(frm_p, from_=0, to=128, textvariable=self.in_extra, width=8).grid(row=0, column=7, padx=4)

        ttk.Label(frm_p, text="Endian").grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(frm_p, text="LE", variable=self.in_endian, value="LE").grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(frm_p, text="BE", variable=self.in_endian, value="BE").grid(row=1, column=2, sticky="w")
        ttk.Checkbutton(frm_p, text="Invert (MONO1→2)", variable=self.in_invert).grid(row=1, column=3, columnspan=2, sticky="w")

        ttk.Checkbutton(frm_p, text="Auto offset = tail − need", variable=self.auto_off).grid(row=2, column=0, columnspan=3, sticky="w")
        ttk.Label(frm_p, text="Offset (abs)").grid(row=2, column=3, sticky="e")
        ttk.Entry(frm_p, textvariable=self.in_offset, width=12).grid(row=2, column=4, sticky="w", padx=4)

        # WL
        frm_wl = ttk.LabelFrame(self, text="Window/Level (яркость/контраст)")
        frm_wl.pack(fill="x", **pad)
        ttk.Checkbutton(frm_wl, text="Auto WL (p1→p99)", variable=self.auto_wl).grid(row=0, column=0, sticky="w")
        ttk.Label(frm_wl, text="Level / Center").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm_wl, textvariable=self.wl_center, width=10).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(frm_wl, text="Width").grid(row=1, column=2, sticky="e")
        ttk.Entry(frm_wl, textvariable=self.wl_width, width=10).grid(row=1, column=3, sticky="w", padx=4)

        # Кнопки
        frm_act = ttk.Frame(self); frm_act.pack(fill="x", **pad)
        ttk.Button(frm_act, text="Конвертировать", command=self.run).pack(side="left")
        ttk.Button(frm_act, text="Остановить", command=self.stop).pack(side="left", padx=6)

        # Прогресс и лог
        self.prog = ttk.Progressbar(self, mode="determinate")
        self.prog.pack(fill="x", padx=10, pady=4)
        self.log = tk.Text(self, height=12, wrap="word")
        self.log.pack(fill="both", expand=True, padx=10, pady=6)

    # --- действия GUI ---
    def pick_file(self):
        p = filedialog.askopenfilename(title="Выберите файл")
        if p:
            self.src_path.set(p)

    def pick_folder(self):
        p = filedialog.askdirectory(title="Выберите папку")
        if p:
            self.src_path.set(p)

    def log_line(self, s: str):
        self.log.insert("end", s + "\n")
        self.log.see("end")
        self.update_idletasks()

    def stop(self):
        self._stop = True

    def run(self):
        src = self.src_path.get().strip()
        if not src:
            messagebox.showwarning("Нет источника", "Выбери файл или папку.")
            return
        path = Path(src)
        files = [path] if path.is_file() else [p for p in path.iterdir() if p.is_file()]
        if not files:
            messagebox.showinfo("Пусто", "Файлов не найдено.")
            return

        self._stop = False
        self.prog["maximum"] = len(files)
        self.prog["value"] = 0
        self.log.delete("1.0", "end")

        self._worker = threading.Thread(target=self._worker_convert, args=(files,), daemon=True)
        self._worker.start()

    def _worker_convert(self, files: list[Path]):
        ok_cnt = fail_cnt = 0
        for i, p in enumerate(files, 1):
            if self._stop:
                self.log_line("Остановлено пользователем.")
                break
            try:
                b = p.read_bytes()
                mv = memoryview(b)
                # offset
                if self.auto_off.get():
                    off = auto_offset(len(b), self.in_rows.get(), self.in_cols.get(), self.in_bits.get(), self.in_extra.get())
                else:
                    off = max(0, int(self.in_offset.get()))

                arr16 = reconstruct_from_raw(
                    mv,
                    rows=self.in_rows.get(),
                    cols=self.in_cols.get(),
                    bits=self.in_bits.get(),
                    extra=self.in_extra.get(),
                    endian=self.in_endian.get(),
                    offset_abs=off,
                    invert_mono1=self.in_invert.get()
                )

                # WL
                if self.auto_wl.get():
                    c, w = auto_wl_p1_p99(arr16)
                else:
                    c, w = float(self.wl_center.get()), float(max(1.0, self.wl_width.get()))

                out_path = p.with_suffix(".dcm")
                write_sc_from_array(arr16, out_path, photometric="MONOCHROME2", wl_center=c, wl_width=w)
                ok_cnt += 1
                self.log_line(f"OK: {p.name} → {out_path.name}  (off={off})")
            except Exception as e:
                fail_cnt += 1
                self.log_line(f"FAIL: {p.name} — {type(e).__name__}: {e}")

            self.prog["value"] = i
            self.update_idletasks()

        self.log_line("-" * 50)
        self.log_line(f"Готово. Успешно: {ok_cnt}  |  Ошибок: {fail_cnt}")

if __name__ == "__main__":
    App().mainloop()
