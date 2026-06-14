"""Confirm birefnet-general actually executes on the NVIDIA GPU via CUDA EP."""
import time
import gpu_init  # noqa: F401  -- must precede onnxruntime; registers CUDA/cuDNN DLL dirs
import onnxruntime as ort

# Make ORT find the CUDA/cuDNN DLLs shipped in the nvidia-*-cu12 pip packages.
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

import fitz
from PIL import Image
from rembg import new_session, remove

PDF = r"D:\E\Data Science Projects\ClaudeDoc\Data\Bad\Bookmark 21.pdf"

print("ORT available providers:", ort.get_available_providers())
session = new_session("birefnet-general",
                      providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
active = session.inner_session.get_providers()
print("Session ACTIVE providers:", active)
if "CUDAExecutionProvider" not in active:
    print("!! CUDA not active — would run on CPU. Aborting.")
    raise SystemExit(1)

doc = fitz.open(PDF)
pix = doc[0].get_pixmap(dpi=200)
img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
doc.close()

# warmup (first call compiles kernels), then time a few
remove(img, session=session)
n = 5
t0 = time.time()
for _ in range(n):
    remove(img, session=session)
dt = (time.time() - t0) / n
print(f"GPU OK. birefnet-general avg {dt*1000:.0f} ms/image at 200 DPI "
      f"({pix.width}x{pix.height}).")
