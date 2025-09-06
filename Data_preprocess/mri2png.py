"""
brats2png_with_masks.py  build 100-slice dataset *with* masks
 Creates   /Group/Modality/<slice-stem>/
   <stem>.png          (windowed grayscale image)
   <stem>_mask.png     (binary whole-tumor mask)
The rest (quotas, uniqueness, full-brain filter) is unchanged from your
previous build_brats100_unique_fullbrain.py
"""
import argparse, csv, json, random
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
QUOTAS = {
    "Healthy_Brain": {"T1": 50, "T2": 50, "FLAIR": 50},
    "Pathological_Brain": {"T1": 50, "T2": 50, "FLAIR": 50},
}
SLICE_MIN, SLICE_MAX = 35, 75
FULL_BRAIN_FRAC = 0.60
MASK_MODALITY = "FLAIR"
OUT_MASK_SUFFIX = "_mask.png"
MOD_SUFFIX = {"T1": "_t1.nii", "T1ce": "_t1ce.nii", "T2": "_t2.nii", "FLAIR": "_flair.nii", "SEG": "_seg.nii"}
random.seed(42)
def find_cases(root: Path, split="Training"):
    base = root / f"MICCAI_BraTS2020_TrainingData"
    case_dirs = []
    for p in sorted(base.glob("BraTS20_*")):
        have = [p.joinpath(p.name + s) for s in MOD_SUFFIX.values()]
        ok = True
        for q in have:
            if not (q.exists() or (q.with_suffix(q.suffix + ".gz")).exists()):
                ok = False
                break
        if ok:
            case_dirs.append(p)
    return case_dirs
def load_nii(path: Path):
    p = path if path.exists() else path.with_suffix(path.suffix + ".gz")
    return nib.load(str(p))
def robust_uint8(img2d):
    lo, hi = np.percentile(img2d, [1, 99])
    if hi <= lo:
        return np.zeros_like(img2d, np.uint8)
    return (np.clip((img2d - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
import numpy as np
import nibabel as nib
from pathlib import Path
def slice_brain_area_map(case_dir: Path) -> np.ndarray:
    """Return an array[Z] with the brainpixel count for each axial slice."""
    flair = load_nii(case_dir / f"{case_dir.name}{MOD_SUFFIX[MASK_MODALITY]}")
    arr = np.asanyarray(flair.dataobj)
    H, W, Z = arr.shape
    areas = np.zeros(Z, dtype=np.int32)
    for z in range(Z):
        sl = arr[..., z].astype(np.float32)
        lo, hi = np.percentile(sl, [1, 99])
        if hi <= lo:
            continue
        sl_norm = np.clip((sl - lo) / (hi - lo), 0, 1)
        try:
            t = threshold_otsu(sl_norm)
        except ValueError:
            continue
        mask = sl_norm > t
        if mask.sum() >= 0.005 * H * W:
            areas[z] = int(mask.sum())
    return areas
def eligible_slices_full_brain(case_dir: Path, z0, z1):
    areas = slice_brain_area_map(case_dir)
    if areas.max() == 0:
        return []
    thr = int(FULL_BRAIN_FRAC * areas.max())
    return [z for z in range(max(0, z0), min(len(areas) - 1, z1) + 1) if areas[z] >= thr]
def gather_candidates(cases, want_pathological, modality):
    out = {}
    for d in cases:
        seg = np.asanyarray(load_nii(d / f"{d.name}{MOD_SUFFIX['SEG']}").dataobj)
        img_shape = seg.shape
        z_keep = eligible_slices_full_brain(d, SLICE_MIN, SLICE_MAX)
        if not z_keep:
            continue
        zs = [z for z in z_keep if ((seg[..., z] > 0).any() if want_pathological else not (seg[..., z] > 0).any())]
        if zs:
            random.shuffle(zs)
            out[d] = zs
    return out
def sample_unique(cases_dict, need, used):
    pool = [(c, z) for c, zs in cases_dict.items() for z in zs if c not in used]
    random.shuffle(pool)
    sel, seen = [], set()
    for c, z in pool:
        if c in seen:
            continue
        sel.append((c, z))
        seen.add(c)
        used.add(c)
        if len(sel) == need:
            break
    return sel
def save_pair(img_vol, seg_arr, z, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(robust_uint8(np.asanyarray(img_vol.dataobj)[..., z])).save(out_dir / f"{stem}.png")
    mask = (seg_arr[..., z] > 0).astype(np.uint8) * 255
    Image.fromarray(mask).save(out_dir / f"{stem}{OUT_MASK_SUFFIX}")
def main(root: str, out_root: str):
    root, out_root = Path(root), Path(out_root)
    cases = find_cases(root)
    if not cases:
        raise SystemExit("No training cases found")
    manifest, counts, used = [], {g: {m: 0 for m in d} for g, d in QUOTAS.items()}, set()
    plan = [
        ("Healthy_Brain", "T1", False),
        ("Healthy_Brain", "T2", False),
        ("Healthy_Brain", "FLAIR", False),
        ("Pathological_Brain", "T1", True),
        ("Pathological_Brain", "T2", True),
        ("Pathological_Brain", "FLAIR", True),
    ]
    for group, mod, want_path in plan:
        picks = sample_unique(gather_candidates(cases, want_path, mod), QUOTAS[group][mod], used)
        for case_dir, z in picks:
            img_vol = load_nii(case_dir / f"{case_dir.name}{MOD_SUFFIX[mod]}")
            seg_arr = np.asanyarray(load_nii(case_dir / f"{case_dir.name}{MOD_SUFFIX['SEG']}").dataobj)
            sub = f"{group}/{mod.replace('T1','T1')}"
            stem = f"{case_dir.name}_{mod}_z{z:03d}"
            save_pair(img_vol, seg_arr, z, out_root / sub / stem, stem)
            manifest.append(
                {
                    "folder": f"{sub}/{stem}",
                    "group": group,
                    "modality": "T1" if mod == "T1" and group.startswith("Patho") else mod,
                    "case": case_dir.name,
                    "slice": z,
                }
            )
            counts[group][mod] += 1
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest[0].keys())
        w.writeheader()
        w.writerows(manifest)
    with open(out_root / "build_config.json", "w") as f:
        json.dump(
            {"SLICE_MIN": SLICE_MIN, "SLICE_MAX": SLICE_MAX, "FULL_BRAIN_FRAC": FULL_BRAIN_FRAC, "counts": counts},
            f,
            indent=2,
        )
    print("Done\n counts:", counts, "\n output root:", out_root)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        help="Path containing BraTS20_Training_Data/",
        default="./DS_dump/1/BraTS2020_TrainingData",
    )
    ap.add_argument("--out", help="Output folder for 100-image dataset", default="./BraTS_DS/")
    args = ap.parse_args()
    main(args.root, args.out)
