from pathlib import Path
import numpy as np
import pandas as pd
from typing import Iterable
from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim
import hashlib
import json, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from typing import Any, Sequence
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
RESULTS = Path("Results_")
CACHE = True
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "Dataset"))
STANDARD_PARAMS = {
    "n_estimators": 250,
    "max_depth": None,
    "random_state": 42,
    "n_jobs": -1,
}
try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except Exception:
    HAS_SMP = False
def resolve_path(p: str | Path) -> Path:
    """Return an absolute path. If `p` is absolute, use it; otherwise join to DATASET_ROOT."""
    p = Path(str(p).strip())
    if p.is_absolute():
        return p
    return DATASET_ROOT / p
def safe_symlink_or_copy(src: Path, dst: Path):
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    except Exception:
        import shutil
        shutil.copy2(src, dst)
def dataset_signature(csv_path: Path, extra: dict | None = None) -> str:
    sig = {"csv_sha1": file_sha1(csv_path)}
    if extra:
        sig.update(extra)
    j = json.dumps(sig, sort_keys=True).encode()
    return hashlib.sha1(j).hexdigest()[:12]
def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
def pick_first_present(df: pd.DataFrame, candidates: Iterable[str], *, required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of the candidate columns found: {list(candidates)}; have={list(df.columns)}")
    return None
def _prep_image_uint8_to_float01(img):
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi > lo:
        x = np.clip((x - lo) / (hi - lo), 0, 1)
    else:
        x /= 255.0
    return x.astype(np.float32, copy=False)
def _to_tensor_nchw(x):
    t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return t
def _adapt_for_model_input(t: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    Ensure the tensor t has the right
    - If model's first conv needs 3 channels and t is 1-ch, repeat to 3.
    - If final input is 3-ch, apply ImageNet normalization (assumes t in [0,1]).
    """
    cin = _first_conv_in_channels(model)
    x = t
    if cin == 3 and x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    if x.shape[1] == 3:
        mean = _IMAGENET_MEAN.to(x.device, x.dtype)
        std = _IMAGENET_STD.to(x.device, x.dtype)
        x = (x - mean) / std
    return x
def _first_conv_in_channels(model: nn.Module) -> int | None:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return m.in_channels
    return None
def _as_logits(y):
    """
    Normalize model output to logits.
    Supports:
      - Pure tensor (logits or probs in [0,1])
      - dict / OrderedDict (e.g. {"out": tensor, ...}); will take the first tensor
    """
    if isinstance(y, (dict, OrderedDict)):
        y = next(iter(y.values()))
    y_min = float(y.min().item())
    y_max = float(y.max().item())
    if 0.0 - 1e-6 <= y_min and y_max <= 1.0 + 1e-6:
        eps = 1e-6
        y = torch.clamp(y, eps, 1.0 - eps)
        return torch.log(y / (1.0 - y))
    return y
def jaccard_index(pred, target, eps=1e-6):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    inter = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - inter
    return (inter + eps) / (union + eps)
def hd95(pred, target, spacing=(1.0, 1.0)):
    d = _surface_distances(pred, target, spacing)
    if not np.isfinite(d).any():
        return float("inf")
    return float(np.percentile(d, 95.0))
def _surface_distances(m1, m2, spacing=(1.0, 1.0)):
    m1 = m1.astype(bool)
    m2 = m2.astype(bool)
    if not (m1.any() and m2.any()):
        return np.array([np.inf])
    m1_edt = ndi.distance_transform_edt(~m1, sampling=spacing)
    m2_edt = ndi.distance_transform_edt(~m2, sampling=spacing)
    m1_surf = np.logical_xor(m1, ndi.binary_erosion(m1))
    m2_surf = np.logical_xor(m2, ndi.binary_erosion(m2))
    d12 = m2_edt[m1_surf]
    d21 = m1_edt[m2_surf]
    return np.concatenate([d12, d21])
def feature_grid(img_tensor, feat_map, max_channels=8):
    """
    Create a HxW grid image from first K channels of feat_map [C,H,W].
    Returns uint8 image suitable for saving with cv2.imwrite.
    """
    C, H, W = feat_map.shape
    K = min(max_channels, C)
    tiles = []
    for c in range(K):
        x = feat_map[c]
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        x = (x * 255.0).astype(np.uint8)
        tiles.append(x)
    rows = []
    cols = 2
    r = []
    for i, t in enumerate(tiles):
        r.append(t)
        if (i + 1) % cols == 0:
            rows.append(np.concatenate(r, axis=1))
            r = []
    if r:
        while len(r) < cols:
            r.append(np.zeros_like(r[0]))
        rows.append(np.concatenate(r, axis=1))
    grid = np.concatenate(rows, axis=0)
    return grid
def linear_cka(X, Y, *, center=True, eps: float = 1e-8) -> float:
    """
    Linear CKA between representations X and Y.
    Accepts numpy arrays or tensors. Returns scalar float.
    Uses (||X^T Y||_F^2) / (||X^T X||_F * ||Y^T Y||_F).
    Dimensions: X:[N, Dx], Y:[N, Dy].
    """
    device = torch.device("cpu")
    X = _to_2d_tensor(X, device=device, dtype=torch.float32)
    Y = _to_2d_tensor(Y, device=device, dtype=torch.float32)
    assert X.shape[0] == Y.shape[0], f"batch mismatch: {X.shape} vs {Y.shape}"
    if center:
        X = _center_columns(X)
        Y = _center_columns(Y)
    XtY = X.T @ Y
    num = (XtY * XtY).sum()
    XtX = X.T @ X
    YtY = Y.T @ Y
    denom = torch.sqrt((XtX * XtX).sum()) * torch.sqrt((YtY * YtY).sum())
    denom = denom + eps
    cka = (num / denom).item()
    return float(cka)
def _to_2d_tensor(x, *, device=None, dtype=torch.float32):
    """
    Convert x (np.ndarray | list | tensor) to a 2D torch.Tensor [N, D].
    - If x is 4D BCHW, apply GAP to [N, C] (more stable than full flatten for CKA).
    - If x is 3D CHW, assume single sample -> [1, C] after GAP.
    - If x is 1D D, make it [1, D].
    - If x is 2D already, keep [N, D].
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if device is not None:
        x = x.to(device)
    x = x.to(dtype)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() == 4:
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
    elif x.dim() > 2:
        x = x.flatten(start_dim=1)
    return x.contiguous()
def _center_columns(X: torch.Tensor) -> torch.Tensor:
    return X - X.mean(dim=0, keepdim=True)
def get_row_value(row, col: str):
    if hasattr(row, col):
        return getattr(row, col)
    try:
        return row[col]
    except Exception:
        raise AttributeError(
            f"Row has no attribute/key '{col}'. Available: {getattr(row, '_fields', None) or list(row.index)}"
        )
def _hr(msg: str | None = None, char: str = "=") -> None:
    """Print a horizontal rule with an optional centred title."""
    width = 80
    if msg:
        pad = (width - len(msg) - 2) // 2
        print(char * pad, msg, "=" * pad)
    else:
        print(char * width)
def _fmt(val: float, precision: int = 4) -> str:
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    return f"{val:.{precision}f}"
def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Return a GitHubflavoured markdown table as string."""
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(header, *rows)]
    def fmt_row(r):
        return " | ".join(str(c).ljust(w) for c, w in zip(r, col_widths))
    out = [fmt_row(header), " | ".join("-" * w for w in col_widths)]
    out += [fmt_row(r) for r in rows]
    return "| " + " |\n| ".join(out) + " |"
def utf8_(directory_path):
    """Process every Python file in the directory and convert them to UTF-8."""
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "rb") as f:
                        content = f.read()
                    print(f"Processing: {file_path}")
                    content_str = content.decode("utf-8", errors="ignore")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content_str)
                    print(f"Successfully converted: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
