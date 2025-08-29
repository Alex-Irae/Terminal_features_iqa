"""
Copy a dataset tree and, in each leaf directory that contains one or more PNGs,
apply distortions to the non-mask PNG(s) (ignore files ending with '_mask.png').
Distortions and severities:
- Gaussian blur: sigma in [1.0, 1.4, 1.8, 2.2, 2.6]
- Gaussian noise: std in [0.02, 0.04, 0.07, 0.10, 0.13] on [0,1]
- JPEG: quality in [80, 50, 30, 10, 1] (artifacts are baked in, output saved as PNG)
- Downsample: scale s in [0.90, 0.75, 0.50, 0.35, 0.20], bicubic back to original size
New MRI-oriented distortions:
- Motion blur (patient movement): kernel length in [5, 9, 13, 17, 21], random angle per image
- Rician noise (MRI magnitude): sigma in [0.02, 0.04, 0.07, 0.10, 0.13]
- Bias field corruption (multiplicative, low-frequency): amplitude in [0.2, 0.35, 0.5, 0.65, 0.8],
  smoothness (Gaussian sigma as fraction of min(H,W)) in [0.03, 0.06, 0.10, 0.14, 0.18]
- Ghosting artifacts (periodic replicas along phase-encode): shift fraction of image in
  [0.01, 0.02, 0.03, 0.05, 0.08] with strengths [0.20, 0.30, 0.40, 0.50, 0.60]
Output naming (default): {stem}_{distortion}_s{severity}.png, e.g. foo_blur_s3.png
Optionally: --name_mode param -> foo_blur_sigma1.6.png, foo_jpeg_q60.png, etc.
"""
import argparse
from pathlib import Path
import shutil
import numpy as np
import cv2
import os
BLUR_SIGMAS = [1.0, 1.4, 1.8, 2.2, 2.6]
NOISE_SIGMAS = [0.02, 0.04, 0.07, 0.10, 0.13]
JPEG_QUALITIES = [80, 50, 30, 10, 1]
DOWNSAMPLE_SCALES = [0.90, 0.75, 0.50, 0.35, 0.20]
MOTION_LENGTHS = [5, 9, 13, 17, 21]
RICIAN_SIGMAS = [0.02, 0.04, 0.07, 0.10, 0.13]
BIAS_AMPLITUDES = [0.20, 0.35, 0.50, 0.65, 0.80]
BIAS_SIGMA_FRACS = [0.03, 0.06, 0.10, 0.14, 0.18]
GHOST_SHIFT_FRACS = [0.01, 0.02, 0.03, 0.05, 0.08]
GHOST_ALPHAS = [0.20, 0.30, 0.40, 0.50, 0.60]
DISTORTION_KEYS = (
    "blur",
    "noise",
    "motion",
    "rician",
    "bias",
    "ghosting",
)
def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return np.ascontiguousarray(img)
    if img.dtype == np.uint16:
        return np.ascontiguousarray((img >> 8).astype(np.uint8))
    if np.issubdtype(img.dtype, np.floating):
        x = np.clip(img, 0.0, 255.0)
        if x.max() <= 1.0:
            x = x * 255.0
        return np.ascontiguousarray(x.round().astype(np.uint8))
    return np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))
def apply_blur(img: np.ndarray, sev_idx: int) -> np.ndarray:
    img = ensure_uint8(img)
    sigma = BLUR_SIGMAS[sev_idx]
    k = int(2 * np.ceil(3.0 * sigma) + 1)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
def apply_noise(img: np.ndarray, sev_idx: int, rng: np.random.Generator) -> np.ndarray:
    img = ensure_uint8(img)
    x = img.astype(np.float32) / 255.0
    std = NOISE_SIGMAS[sev_idx]
    noise = rng.normal(0.0, std, size=x.shape).astype(np.float32)
    y = np.clip(x + noise, 0.0, 1.0)
    return (y * 255.0).round().astype(np.uint8)
def apply_jpeg(img: np.ndarray, sev_idx: int) -> np.ndarray:
    img8 = ensure_uint8(img)
    if img8.ndim == 3 and img8.shape[2] == 4:
        img8 = img8[:, :, :3]
    original_shape = img8.shape
    is_grayscale = img8.ndim == 2
    q = int(JPEG_QUALITIES[sev_idx])
    q = max(1, min(q, 100))
    img8 = np.ascontiguousarray(img8)
    encode_img = img8
    try:
        ok, buf = cv2.imencode(".jpg", encode_img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok and is_grayscale:
            encode_img = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
            ok, buf = cv2.imencode(".jpg", encode_img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            print(f"Warning: JPEG encoding failed for quality {q}, using original image")
            return img8
    except Exception as e:
        print(f"Warning: JPEG encoding exception: {e}, using original image")
        return img8
    try:
        decoded = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            print("Warning: JPEG decoding failed, using original image")
            return img8
        if is_grayscale:
            if decoded.ndim == 3:
                decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        else:
            if len(original_shape) == 3 and original_shape[2] == 3 and decoded.ndim == 2:
                decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
        return ensure_uint8(decoded)
    except Exception as e:
        print(f"Warning: JPEG decoding exception: {e}, using original image")
        return img8
def apply_downsample(img: np.ndarray, sev_idx: int) -> np.ndarray:
    img = ensure_uint8(img)
    s = float(DOWNSAMPLE_SCALES[sev_idx])
    h, w = img.shape[:2]
    new_h = max(1, int(round(h * s)))
    new_w = max(1, int(round(w * s)))
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return out
def _motion_psf(k: int, angle_deg: float) -> np.ndarray:
    k = int(k)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    psf = np.zeros((k, k), dtype=np.float32)
    psf[k // 2, :] = 1.0
    psf /= psf.sum()
    M = cv2.getRotationMatrix2D((k / 2.0 - 0.5, k / 2.0 - 0.5), angle_deg, 1.0)
    rot = cv2.warpAffine(psf, M, (k, k), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    s = rot.sum()
    if s > 0:
        rot /= s
    return rot
def apply_motion(img: np.ndarray, sev_idx: int, rng: np.random.Generator) -> np.ndarray:
    img = ensure_uint8(img)
    k = int(MOTION_LENGTHS[sev_idx])
    angle = float(rng.uniform(0.0, 180.0))
    psf = _motion_psf(k, angle)
    if img.ndim == 2:
        return cv2.filter2D(img, -1, psf, borderType=cv2.BORDER_REFLECT_101)
    else:
        chs = []
        for c in range(img.shape[2]):
            chs.append(cv2.filter2D(img[:, :, c], -1, psf, borderType=cv2.BORDER_REFLECT_101))
        return np.stack(chs, axis=2)
def apply_rician(img: np.ndarray, sev_idx: int, rng: np.random.Generator) -> np.ndarray:
    img8 = ensure_uint8(img)
    x = img8.astype(np.float32) / 255.0
    sigma = float(RICIAN_SIGMAS[sev_idx])
    n1 = rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    n2 = rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    y = np.sqrt(np.maximum((x + n1) ** 2 + n2**2, 0.0))
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).round().astype(np.uint8)
def apply_bias_field(img: np.ndarray, sev_idx: int, rng: np.random.Generator) -> np.ndarray:
    img8 = ensure_uint8(img)
    x = img8.astype(np.float32) / 255.0
    h, w = x.shape[:2]
    amp = float(BIAS_AMPLITUDES[sev_idx])
    sigma_frac = float(BIAS_SIGMA_FRACS[sev_idx])
    sigma = max(1.0, sigma_frac * min(h, w))
    g = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    g = cv2.GaussianBlur(g, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    g_min, g_max = float(g.min()), float(g.max())
    if g_max - g_min < 1e-6:
        g01 = np.zeros_like(g, dtype=np.float32) + 0.5
    else:
        g01 = (g - g_min) / (g_max - g_min)
    field = 1.0 + (g01 - 0.5) * 2.0 * amp
    if x.ndim == 3:
        field = field[:, :, None]
    y = np.clip(x * field, 0.0, 1.0)
    return (y * 255.0).round().astype(np.uint8)
def _shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    if img.ndim == 2:
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    else:
        chs = []
        for c in range(img.shape[2]):
            chs.append(
                cv2.warpAffine(img[:, :, c], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            )
        return np.stack(chs, axis=2)
def apply_ghosting(img: np.ndarray, sev_idx: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate MRI ghosting by adding several attenuated shifted replicas along a (random) phase-encode axis.
    We keep overall brightness stable by normalizing by total weight.
    """
    img8 = ensure_uint8(img)
    x = img8.astype(np.float32) / 255.0
    h, w = x.shape[:2]
    axis = int(rng.integers(0, 2))
    shift_frac = float(GHOST_SHIFT_FRACS[sev_idx])
    alpha_base = float(GHOST_ALPHAS[sev_idx])
    num_ghosts = 2 + sev_idx // 2
    if axis == 0:
        base_shift = max(1, int(round(h * shift_frac)))
        dx, dy = 0, base_shift
    else:
        base_shift = max(1, int(round(w * shift_frac)))
        dx, dy = base_shift, 0
    y = x.copy()
    weights = [1.0]
    for i in range(1, num_ghosts + 1):
        w_i = alpha_base * (0.6 ** (i - 1))
        ghost = _shift_image(x, dx * i, dy * i)
        y += w_i * ghost
        weights.append(w_i)
    total_w = float(np.sum(weights))
    y = np.clip(y / max(total_w, 1e-6), 0.0, 1.0)
    return (y * 255.0).round().astype(np.uint8)
def read_image(path: Path) -> np.ndarray:
    """Read PNG preserving channels."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img
def save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), arr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")
def copy_tree(src_root: Path, dst_root: Path) -> None:
    """
    Copy files and directories from src_root into dst_root, preserving structure.
    Uses copy2 to preserve metadata.
    """
    for root, dirs, files in os.walk(src_root):
        rel = Path(root).relative_to(src_root)
        target_dir = dst_root / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (target_dir / d).mkdir(exist_ok=True)
        for f in files:
            src = Path(root) / f
            dst = target_dir / f
            shutil.copy2(src, dst)
def distort_one(img: np.ndarray, distortion: str, sev_idx: int, rng: np.random.Generator) -> np.ndarray:
    if distortion == "blur":
        return apply_blur(img, sev_idx)
    elif distortion == "noise":
        return apply_noise(img, sev_idx, rng)
    elif distortion == "jpeg":
        return apply_jpeg(img, sev_idx)
    elif distortion == "downsample":
        return apply_downsample(img, sev_idx)
    elif distortion == "motion":
        return apply_motion(img, sev_idx, rng)
    elif distortion == "rician":
        return apply_rician(img, sev_idx, rng)
    elif distortion == "bias":
        return apply_bias_field(img, sev_idx, rng)
    elif distortion == "ghosting":
        return apply_ghosting(img, sev_idx, rng)
    else:
        raise ValueError(f"Unknown distortion: {distortion}")
def build_name(stem: str, distortion: str, sev_idx: int, name_mode: str) -> str:
    """Return output filename (without extension)."""
    sev = sev_idx + 1
    if name_mode == "index":
        return f"{stem}_{distortion}_s{sev}"
    elif name_mode == "param":
        if distortion == "blur":
            return f"{stem}_blur_sigma{BLUR_SIGMAS[sev_idx]}"
        if distortion == "noise":
            return f"{stem}_noise_sigma{NOISE_SIGMAS[sev_idx]}"
        if distortion == "jpeg":
            return f"{stem}_jpeg_q{JPEG_QUALITIES[sev_idx]}"
        if distortion == "downsample":
            return f"{stem}_downsample_s{DOWNSAMPLE_SCALES[sev_idx]}"
        if distortion == "motion":
            return f"{stem}_motion_k{MOTION_LENGTHS[sev_idx]}"
        if distortion == "rician":
            return f"{stem}_rician_sigma{RICIAN_SIGMAS[sev_idx]}"
        if distortion == "bias":
            return f"{stem}_bias_amp{BIAS_AMPLITUDES[sev_idx]}_sf{BIAS_SIGMA_FRACS[sev_idx]}"
        if distortion == "ghosting":
            return f"{stem}_ghosting_shiftf{GHOST_SHIFT_FRACS[sev_idx]}_a{GHOST_ALPHAS[sev_idx]}"
        return f"{stem}_{distortion}_s{sev}"
    else:
        raise ValueError(f"Invalid name_mode: {name_mode}")
def is_mask_png(p: Path) -> bool:
    return p.suffix.lower() == ".png" and p.stem.endswith("_mask")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_root", type=str, required=True, help="Input dataset root to copy.")
    ap.add_argument("--out_root", type=str, required=True, help="Output root (copied tree + added distortions).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for severity and noise.")
    ap.add_argument(
        "--name_mode",
        type=str,
        choices=("index", "param"),
        default="index",
        help="File naming: 'index' -> _s{1..5} (default), 'param' -> embeds numeric parameter.",
    )
    ap.add_argument(
        "--random",
        action="store_true",
        help="If set, pick one random severity per distortion type; otherwise generate all severities 1..5.",
    )
    ap.add_argument(
        "--folder",
        action="store_true",
        help=(
            "If set, and originals are flat in one directory (no per-image leaf folders), "
            "create a subfolder per image (named after the image stem) in the OUT tree, and place "
            "the original + its mask (if found) + all distorted variants inside it. "
            "If the current directory name already equals the image stem, do NOT nest another folder."
        ),
    )
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help=("Optional comma-separated subset of distortions to generate. " "Choices: " + ",".join(DISTORTION_KEYS)),
    )
    args = ap.parse_args()
    base_root = Path(args.base_root).resolve()
    out_root = Path(args.out_root).resolve()
    rng = np.random.default_rng(args.seed)
    if not base_root.exists():
        raise FileNotFoundError(f"Base root not found: {base_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Copying tree from {base_root} to {out_root} ...")
    copy_tree(base_root, out_root)
    print("Copy complete. Adding distortions...")
    if args.only:
        selected = [d.strip().lower() for d in args.only.split(",") if d.strip()]
        for d in selected:
            if d not in DISTORTION_KEYS:
                raise ValueError(f"--only contains unknown distortion '{d}'. Valid: {DISTORTION_KEYS}")
        distortion_list = tuple(selected)
    else:
        distortion_list = DISTORTION_KEYS
    total_imgs = 0
    total_written = 0
    for root, _, files in os.walk(base_root):
        rel = Path(root).relative_to(base_root)
        dst_dir_parent = out_root / rel
        pngs = [Path(root) / f for f in files if f.lower().endswith(".png")]
        non_masks = [p for p in pngs if not is_mask_png(p)]
        if not non_masks:
            continue
        for src_png in non_masks:
            img = read_image(src_png)
            stem = src_png.stem
            total_imgs += 1
            already_leaf = Path(root).name == stem
            if args.folder and not already_leaf:
                dst_dir = dst_dir_parent / stem
                dst_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(src_png, dst_dir / src_png.name)
                except Exception as e:
                    print(f"[WARN] Failed to copy original {src_png} -> {dst_dir}: {e}")
                mask_src = Path(root) / f"{stem}_mask.png"
                if mask_src.exists():
                    try:
                        shutil.copy2(mask_src, dst_dir / mask_src.name)
                    except Exception as e:
                        print(f"[WARN] Failed to copy mask {mask_src} -> {dst_dir}: {e}")
            else:
                dst_dir = dst_dir_parent
            for distortion in distortion_list:
                sev_indices = [int(rng.integers(0, 5))] if args.random else list(range(5))
                for sev_idx in sev_indices:
                    out_img = distort_one(img, distortion, sev_idx, rng)
                    out_name = build_name(stem, distortion, sev_idx, args.name_mode) + ".png"
                    save_png(dst_dir / out_name, out_img)
                    total_written += 1
        if total_imgs % 50 == 0 and total_imgs > 0:
            print(f"Processed {total_imgs} images...")
    print(f"Done. Base images found: {total_imgs}. Distorted files written: {total_written}.")
    print(f"Output root: {out_root}")
if __name__ == "__main__":
    main()
