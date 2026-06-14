"""
Import this BEFORE onnxruntime to make the CUDA/cuDNN DLLs from the nvidia-*-cu12
pip packages discoverable. cuDNN's main DLL loads its engine sublibraries by bare
name, so the nvidia\\*\\bin folders must be on the DLL search path or you get
"Could not locate cudnn_engines_tensor_ir64_9.dll" and a silent fall back to CPU.
"""
import os
from pathlib import Path


def enable_cuda_dlls() -> list[str]:
    added = []
    try:
        import nvidia
    except Exception as e:  # nvidia pip packages not installed
        print("gpu_init: nvidia runtime packages not found:", e)
        return added
    nv_root = Path(nvidia.__path__[0])
    for sub in sorted(nv_root.iterdir()):
        bin_dir = sub / "bin"
        if bin_dir.is_dir():
            try:
                os.add_dll_directory(str(bin_dir))
            except Exception:
                pass
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            added.append(str(bin_dir))
    return added


_ADDED = enable_cuda_dlls()
