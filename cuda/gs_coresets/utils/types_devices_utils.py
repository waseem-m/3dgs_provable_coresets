# gs_coresets.utils/types_devices_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional, Sequence, Tuple, Union
import os
import threading

import numpy as np
import torch


# =============================================================================
# Device resolution
# =============================================================================

# Accept: None | "auto" | "default" | "cpu" | "cuda" | "cuda:1" | "mps" | int (cuda index) | torch.device
DeviceInput = Optional[Union[str, int, torch.device]]

def _autodetect_device() -> torch.device:
    """
    Automatic device policy:
      1) Environment overrides (GS_DEVICE / PYTORCH_DEVICE)
      2) CUDA (cuda:0 respecting CUDA_VISIBLE_DEVICES)
      3) Apple MPS
      4) CPU
    """
    env_dev = os.getenv("GS_DEVICE") or os.getenv("PYTORCH_DEVICE")
    if env_dev:
        return resolve_torch_device(env_dev)

    if torch.cuda.is_available():
        # pick the first visible GPU; caller can re-init to another index if needed.
        return torch.device("cuda:0")

    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass

    return torch.device("cpu")


def resolve_torch_device(
    device: DeviceInput = None,
    *,
    prefer_cuda: bool = True,
    index: int = 0
) -> torch.device:
    """
    Resolve a device spec to a torch.device.

    Parameters
    ----------
    device : DeviceInput
        - torch.device("cuda:0"), "cuda:0", "cpu", "mps", etc.
        - "auto"/"default"/None -> auto policy
        - int -> interpreted as CUDA index if CUDA is available, else CPU
    prefer_cuda : bool
        When device is None/"auto" and both CUDA and MPS are available, prefers CUDA.
        (Kept for backward compatibility; currently only affects default index selection.)
    index : int
        Default CUDA index when device is None and CUDA is available.

    Returns
    -------
    torch.device
    """
    # torch.device passed as-is (guard for CUDA/MPS availability)
    if isinstance(device, torch.device):
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        if device.type == "mps":
            try:
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    return device
            except Exception:
                return torch.device("cpu")
        return device

    # Allow an integer CUDA index (fallback to CPU if no CUDA)
    if isinstance(device, int):
        return torch.device(f"cuda:{int(device)}") if torch.cuda.is_available() else torch.device("cpu")

    # String handling
    if isinstance(device, str):
        s = device.lower().strip()
        if s in {"auto", "default"}:
            device = None  # fall through to auto
        elif s == "cpu":
            return torch.device("cpu")
        elif s == "cuda":
            return torch.device(f"cuda:{index}") if torch.cuda.is_available() else torch.device("cpu")
        elif s.startswith("cuda:"):
            try:
                idx = int(s.split(":", 1)[1])
            except ValueError:
                idx = 0
            return torch.device(f"cuda:{idx}") if torch.cuda.is_available() else torch.device("cpu")
        elif s == "mps":
            try:
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    return torch.device("mps")
            except Exception:
                pass
            return torch.device("cpu")
        else:
            # Best-effort: let torch.device parse; guard CUDA availability.
            try:
                dev_obj = torch.device(s)
                if dev_obj.type == "cuda" and not torch.cuda.is_available():
                    return torch.device("cpu")
                return dev_obj
            except Exception:
                # fall through to auto
                device = None

    # Auto policy (env -> CUDA -> MPS -> CPU)
    if device is None:
        dev = _autodetect_device()
        if dev.type == "cuda" and prefer_cuda:
            # ensure the default cuda index can be requested via 'index' arg
            return torch.device(f"cuda:{index}") if torch.cuda.is_available() else torch.device("cpu")
        return dev

    # Fallback
    return torch.device("cpu")


def resolve_device_name(
    device: DeviceInput = None,
    *,
    prefer_cuda: bool = True,
    index: int = 0
) -> str:
    """Convenience wrapper that returns the resolved device as a string (e.g., 'cuda:0', 'cpu')."""
    return str(resolve_torch_device(device, prefer_cuda=prefer_cuda, index=index))


# =============================================================================
# Type introspection
# =============================================================================

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float], Sequence[Sequence[float]]]

def is_torch_tensor(x: Any) -> bool:
    return isinstance(x, torch.Tensor)

def is_numpy_array(x: Any) -> bool:
    return isinstance(x, np.ndarray)


# =============================================================================
# Singleton-like Torch device/dtype context (no module-level dtype/eps)
# =============================================================================

@dataclass(frozen=True, slots=True)
class TorchDeviceContext:
    """
    Immutable holder for the project's default torch device & dtype, plus derived constants.

    Access the project dtype/eps ONLY via this instance:
        from gs_coresets.utils.types_devices_utils import torch_ctx

        torch_ctx.dtype
        torch_ctx.denom_eps

        x = torch.zeros(3, device=torch_ctx.device, dtype=torch_ctx.dtype)
        y = torch_ctx.as_tensor([1,2,3])                # device=torch_ctx.device, dtype=torch_ctx.dtype
        z = torch_ctx.to(y)                              # move/cast to ctx device (keeps dtype unless provided)

    Re-initialize early in your entry-point if you want a specific device/dtype:
        TorchDeviceContext.init("cuda:1", dtype=torch.float16)   # will update eps accordingly
    """
    device: torch.device
    _TORCH_FLOAT: torch.dtype
    _FLOAT_DENOM_EPS: float

    _instance: ClassVar[Optional["TorchDeviceContext"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    # --- lifecycle -----------------------------------------------------------
    @classmethod
    def init(cls, device: DeviceInput = None, dtype: Optional[torch.dtype] = None) -> "TorchDeviceContext":
        """
        Create and install a new default context. Safe to call multiple times; replaces the singleton.
        Do this EARLY in your program to avoid surprising logs/import-time behavior.
        """
        dt = dtype or torch.float32
        inst = cls(
            device=resolve_torch_device(device),
            _TORCH_FLOAT=dt,
            _FLOAT_DENOM_EPS=float(torch.finfo(dt).eps),
        )
        with cls._lock:
            cls._instance = inst
        return inst

    @classmethod
    def get(cls) -> "TorchDeviceContext":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    dt = torch.float32
                    cls._instance = cls(
                        device=resolve_torch_device(None),
                        _TORCH_FLOAT=dt,
                        _FLOAT_DENOM_EPS=float(torch.finfo(dt).eps),
                    )
        return cls._instance

    # --- convenience helpers ------------------------------------------------
    @property
    def dtype(self) -> torch.dtype:
        """Alias for code that expects 'ctx.dtype'."""
        return self._TORCH_FLOAT

    @property
    def denom_eps(self) -> float:
        """Alias for code that expects 'ctx.denom_eps'."""
        return self._FLOAT_DENOM_EPS


    def make_generator(self, seed: Optional[int] = None) -> Optional[torch.Generator]:
        """
        Create a torch.Generator on this context's device when supported.
        Returns None if seed is None.
        """
        if seed is None:
            return None
        try:
            g = torch.Generator(device=self.device)  # newer PyTorch
        except TypeError:
            g = torch.Generator()                    # older PyTorch
        g.manual_seed(int(seed))
        return g

    def to(self, t: torch.Tensor, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Move/cast an existing tensor onto this context (preserving dtype unless provided)."""
        return t.to(device=self.device, dtype=(dtype or t.dtype))

    def as_tensor(self, data, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Create a tensor on this context."""
        return torch.as_tensor(data, device=self.device, dtype=(dtype or self._TORCH_FLOAT))


# Module-level accessor (import as `torch_ctx`)
torch_ctx: TorchDeviceContext = TorchDeviceContext.get()


# =============================================================================
# Conversions: numpy <-> torch, lists (defaulting to torch_ctx.*)
# =============================================================================

def to_numpy(x: ArrayLike, *, dtype: Optional[np.dtype] = None, copy: bool = False) -> np.ndarray:
    """
    Convert torch/np/list-like to np.ndarray on CPU.
    - torch.Tensor: detach->cpu->numpy (copy only if requested).
    - list-like: np.asarray (copy controlled by flag).
    """
    if is_numpy_array(x):
        arr = x
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        if copy:
            arr = np.array(arr, dtype=arr.dtype, copy=True)
        return arr

    if is_torch_tensor(x):
        arr = x.detach().cpu().numpy()
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        if copy:
            arr = np.array(arr, dtype=arr.dtype, copy=True)
        return arr

    arr = np.asarray(x)
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    if copy:
        arr = np.array(arr, dtype=arr.dtype, copy=True)
    return arr


def to_torch(
    x: ArrayLike,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    copy: bool = False,
) -> torch.Tensor:
    """
    Convert np/torch/list-like to torch.Tensor on the requested device/dtype.
    Defaults: device=torch_ctx.device, dtype=torch_ctx.TORCH_FLOAT
    """
    dev = device or torch_ctx.device
    dt = dtype or torch_ctx.dtype

    if is_torch_tensor(x):
        t = x
        if (dt is not None and t.dtype != dt) or (dev is not None and t.device != dev):
            t = t.to(device=dev or t.device, dtype=dt or t.dtype)
        if copy:
            t = t.clone()
        return t

    if is_numpy_array(x):
        arr = x
        if copy:
            arr = np.array(arr, copy=True)
        t = torch.from_numpy(arr)
        if dt is not None or dev is not None:
            t = t.to(device=dev or t.device, dtype=dt or t.dtype)
        return t

    # list-like / scalar
    return torch.tensor(x, dtype=dt, device=dev)


def same_device_dtype(x: torch.Tensor, like: torch.Tensor, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Move/cast `x` to match `like` (device + dtype). If dtype is given, it overrides."""
    return x.to(device=like.device, dtype=dtype or like.dtype)


# =============================================================================
# Math helpers: normalization
# =============================================================================

def normalize_vec3_torch(v: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Normalize last-dim=3 vectors in torch, preserving shape and dtype/device."""
    assert v.shape[-1] == 3, f"Expected last dim = 3, got {tuple(v.shape)}"
    e = torch.tensor(eps if eps is not None else torch_ctx.denom_eps,
                     device=v.device, dtype=v.dtype)
    n = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(e)
    return v / n


def normalize_vec3_numpy(v: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Normalize last-dim=3 vectors in numpy, preserving shape and dtype."""
    assert v.shape[-1] == 3, f"Expected last dim = 3, got {tuple(v.shape)}"
    e = eps if eps is not None else float(np.finfo(np.float32).eps)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, e)
    return v / n


# =============================================================================
# Shapes: 3x4 / 4x4 transforms (row-major)
# =============================================================================

def _as_2d_numpy(x: ArrayLike) -> np.ndarray:
    arr = to_numpy(x)
    if arr.ndim == 1:
        # Allow flat 12 or 16
        if arr.size in (12, 16):
            arr = arr.reshape(3, 4) if arr.size == 12 else arr.reshape(4, 4)
        else:
            raise ValueError(f"Cannot reshape 1D array of length {arr.size} into 3x4/4x4.")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {arr.shape}.")
    return arr


def as_4x4_np(m: ArrayLike, *, pad_if_3x4: bool = True) -> np.ndarray:
    """
    Convert 3x4 or 4x4 input (torch/np/list) into a (4,4) float32 numpy array.

    If input is 3x4 and pad_if_3x4=True -> returns [[R|t],[0 0 0 1]].
    If input is 4x4 -> returned as-is (cast to float32).
    Accepted shapes: (3,4), (4,4), or flat length 12/16.
    """
    A = _as_2d_numpy(m)
    if A.shape == (4, 4):
        out = A.astype(np.float32, copy=False)
        return out
    if A.shape == (3, 4):
        if not pad_if_3x4:
            raise ValueError("Input is 3x4 and pad_if_3x4=False.")
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = A.astype(np.float32, copy=False)
        return out
    raise ValueError(f"Unsupported matrix shape {A.shape}; expected (3,4) or (4,4).")


def as_3x4_np(m: ArrayLike) -> np.ndarray:
    """Convert 3x4 or 4x4 input into a (3,4) float32 numpy array by dropping the last row if needed."""
    A = _as_2d_numpy(m).astype(np.float32, copy=False)
    if A.shape == (3, 4):
        return A
    if A.shape == (4, 4):
        return A[:3, :]
    raise ValueError(f"Unsupported matrix shape {A.shape}; expected (3,4) or (4,4).")


def as_4x4_torch(
    m: ArrayLike,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """3x4/4x4 -> (4,4) torch tensor on device/dtype (defaults to torch_ctx.*)."""
    dt = dtype or torch_ctx.dtype
    dev = device or torch_ctx.device
    return to_torch(as_4x4_np(m), device=dev, dtype=dt)


def as_3x4_torch(
    m: ArrayLike,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """3x4/4x4 -> (3,4) torch tensor on device/dtype (defaults to torch_ctx.*)."""
    dt = dtype or torch_ctx.dtype
    dev = device or torch_ctx.device
    return to_torch(as_3x4_np(m), device=dev, dtype=dt)


def as_4x4_list(m: ArrayLike) -> List[List[float]]:
    """Same as as_4x4_np(...) but returns a nested Python list (float) for JSON serialization."""
    return as_4x4_np(m).tolist()


def as_3x4_list(m: ArrayLike) -> List[List[float]]:
    """Same as as_3x4_np(...) but returns a nested Python list (float)."""
    return as_3x4_np(m).tolist()


# =============================================================================
# 3-vector & dims helpers
# =============================================================================

def as_vec3_numpy(v: Union[np.ndarray, torch.Tensor, Tuple[float, float, float]],
                  *,
                  dtype: np.dtype = np.float64) -> np.ndarray:
    """Return a NumPy (3,) vector, cast to dtype (default float64)."""
    return to_numpy(v, dtype=dtype).reshape(3)


def as_vec3_torch(v: Union[np.ndarray, torch.Tensor, Tuple[float, float, float]],
                  *,
                  device: Optional[torch.device] = None,
                  dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Return a Torch (3,) vector on device/dtype (defaults to torch_ctx.*)."""
    dt = dtype or torch_ctx.dtype
    dev = device or torch_ctx.device
    return to_torch(v, device=dev, dtype=dt).reshape(3)


def as_dims3(dims: Union[Tuple[int, int, int], Sequence[int]]) -> Tuple[int, int, int]:
    """Validate/cast dims to (nx, ny, nz) ints."""
    if len(dims) != 3:
        raise AssertionError("dims must be a length-3 sequence: (nx, ny, nz)")
    return int(dims[0]), int(dims[1]), int(dims[2])


# =============================================================================
# Small Python list helpers (for JSON)
# =============================================================================

def to_py_list_floats(x: ArrayLike) -> List[float]:
    """Flatten to a 1D list of Python floats. Useful for JSON dumps of small vectors."""
    arr = to_numpy(x, dtype=np.float32).reshape(-1)
    return [float(v) for v in arr]

def to_py_nested_list(x: ArrayLike) -> List[List[float]]:
    """2D numeric -> nested list (float)."""
    arr = to_numpy(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    return arr.tolist()


__all__ = [
    # device
    "DeviceInput", "resolve_torch_device", "resolve_device_name",
    # dtype/device helpers (use torch_ctx.* for dtype/eps)
    "is_torch_tensor", "is_numpy_array", "to_numpy", "to_torch", "same_device_dtype",
    # 3x4 / 4x4 transforms
    "as_4x4_np", "as_3x4_np", "as_4x4_torch", "as_3x4_torch", "as_4x4_list", "as_3x4_list",
    # 3-vector & dims helpers
    "as_vec3_numpy", "as_vec3_torch", "as_dims3",
    # lists
    "to_py_list_floats", "to_py_nested_list",
    # singleton context (the only source for dtype/eps)
    "TorchDeviceContext", "torch_ctx",
]
