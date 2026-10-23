"""Inspect Python, PyTorch, CUDA and the GPU, and pick the inference device."""

from __future__ import annotations

import logging
import platform
import sys
from dataclasses import asdict, dataclass, field

log = logging.getLogger(__name__)


class DeviceError(RuntimeError):
    """The requested device cannot be used."""


@dataclass
class HardwareInfo:
    python: str = platform.python_version()
    platform: str = platform.platform()
    torch: str | None = None
    torch_cuda_build: str | None = None
    cuda_available: bool = False
    cuda_works: bool = False
    gpu_name: str | None = None
    gpu_capability: str | None = None
    vram_total_mb: float | None = None
    vram_free_mb: float | None = None
    opencv: str | None = None
    ultralytics: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        lines = [
            f"Python {self.python} on {self.platform}",
            f"PyTorch {self.torch or 'not installed'} (CUDA build: {self.torch_cuda_build or 'none / CPU-only'})",
        ]
        if self.gpu_name:
            lines.append(
                f"GPU: {self.gpu_name} (compute {self.gpu_capability}), "
                f"VRAM {self.vram_free_mb:.0f} MB free / {self.vram_total_mb:.0f} MB total"
            )
        lines.append(f"CUDA usable: {'yes' if self.cuda_works else 'no'}")
        lines.append(f"OpenCV {self.opencv or 'not installed'}, Ultralytics {self.ultralytics or 'not installed'}")
        lines.extend(f"note: {n}" for n in self.notes)
        return "\n".join(lines)


def inspect_hardware() -> HardwareInfo:
    info = HardwareInfo()
    try:
        import cv2

        info.opencv = cv2.__version__
    except ImportError:
        info.notes.append("OpenCV is not installed")
    try:
        import ultralytics

        info.ultralytics = ultralytics.__version__
    except ImportError:
        info.notes.append("Ultralytics is not installed")
    try:
        import torch
    except ImportError:
        info.notes.append("PyTorch is not installed")
        return info

    info.torch = torch.__version__
    info.torch_cuda_build = torch.version.cuda
    info.cuda_available = torch.cuda.is_available()
    if not info.cuda_available:
        if info.torch_cuda_build is None:
            info.notes.append("This PyTorch build is CPU-only. Install a CUDA build to use the GPU.")
        else:
            info.notes.append("PyTorch has CUDA support but no usable GPU/driver was found.")
        return info

    try:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.gpu_name = props.name
        info.gpu_capability = f"{props.major}.{props.minor}"
        free, total = torch.cuda.mem_get_info(idx)
        info.vram_free_mb = free / 2**20
        info.vram_total_mb = total / 2**20
        arch = f"sm_{props.major}{props.minor}"
        supported = torch.cuda.get_arch_list()
        if supported and arch not in supported and not any(a.startswith("compute_") for a in supported):
            info.notes.append(f"{arch} is not in this PyTorch build's arch list {supported}")
        # A real kernel launch is the only reliable check that CUDA works.
        x = torch.randn(256, 256, device="cuda")
        _ = (x @ x).sum().item()
        torch.cuda.synchronize()
        info.cuda_works = True
    except Exception as exc:  # driver/arch mismatches surface here
        info.notes.append(f"CUDA test kernel failed: {exc}")
    return info


def select_device(requested: str, half_setting: str, hw: HardwareInfo) -> tuple[str, bool]:
    """Return (device, use_half). Raises DeviceError if CUDA was explicitly requested but is unusable."""
    req = requested.strip().lower()
    if req == "cpu":
        device = "cpu"
    elif req.startswith("cuda"):
        if not hw.cuda_works:
            raise DeviceError(
                f"device '{requested}' was requested but CUDA is not usable:\n{hw.describe()}\n"
                "Install a CUDA-enabled PyTorch build (see README) or set device: cpu."
            )
        device = "cuda:0" if req == "cuda" else req
    else:
        device = "cuda:0" if hw.cuda_works else "cpu"
        if device == "cpu":
            log.warning("Running on CPU. Expect low frame rates; see README > GPU setup.")
    if device == "cpu":
        use_half = False
        if half_setting == "on":
            log.warning("FP16 was requested but is not used on CPU.")
    else:
        use_half = half_setting in ("auto", "on")
    return device, use_half


if __name__ == "__main__":  # pragma: no cover - convenience
    print(inspect_hardware().describe())
    sys.exit(0)
