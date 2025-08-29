import argparse
import csv
import re
from pathlib import Path
DISTORTIONS = ("blur", "noise", "jpeg", "downsample",'rician','ghosting','motion','bias')
FNAME_RE = re.compile(r"^(?P<base_id>.+)_(?P<distortion>blur|noise|jpeg|downsample|ghosting|rician|motion|bias|)_s(?P<severity>[1-5])\.png$", re.IGNORECASE)
ZIDX_RE = re.compile(r"(z\d{1,4})$", re.IGNORECASE)
def find_mask_in_leaf(leaf_dir: Path) -> Path | None:
    candidates = sorted(p for p in leaf_dir.glob("*.png") if p.name.lower().endswith("_mask.png"))
    if not candidates:
        return None
    return candidates[0]
def parse_row(img_path: Path, root: Path) -> dict | None:
    """
    Parse a distorted image path into the requested fields.
    Expected structure (example):
      root/Healthy_Brain/FLAIR/BraTS20_Training_032_FLAIR_z045/BraTS20_Training_032_FLAIR_z045_downsample_s3.png
    - base_id: BraTS20_Training_032_FLAIR_z045
    - modality: parent directory of the leaf (e.g., FLAIR)
    - health: grandparent directory of the leaf (e.g., Healthy_Brain)
    - z_index: last token in base_id that matches r"z\\d+" (e.g., z045)
    - distortion: from file name
    - severity: integer 1..5
    - path_img: full path to the distorted image
    - path_mask_original: full path to *_mask.png inside the leaf directory
    """
    m = FNAME_RE.match(img_path.name)
    if not m:
        return None
    base_id = m.group("base_id")
    distortion = m.group("distortion").lower()
    severity = int(m.group("severity"))
    leaf_dir = img_path.parent
    modality_dir = leaf_dir.parent
    modality = modality_dir.name
    health_dir = modality_dir.parent
    health = health_dir.name
    z_match = ZIDX_RE.search(base_id)
    z_index = z_match.group(1) if z_match else ""
    mask_path = find_mask_in_leaf(leaf_dir)
    return {
        "base_id": base_id,
        "modality": modality,
        "health": health,
        "z_index": z_index,
        "distortion": distortion,
        "severity": severity,
        "path_img": str(img_path),
        "path_mask_original": str(mask_path) if mask_path is not None else "",
    }
def main():
    ap = argparse.ArgumentParser(description="Export CSV index for distorted dataset.")
    ap.add_argument("--root", type=str, required=True, help="Root of the distorted dataset to index.")
    ap.add_argument("--out_csv", type=str, required=True, help="Path to write the CSV file.")
    ap.add_argument("--include_originals", action="store_true",
                    help="If set, also index images without _<distortion>_sN (distortion='', severity='').")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out_csv = Path(args.out_csv).resolve()
    rows: list[dict] = []
    for p in sorted(root.rglob("*.png")):
        if p.name.lower().endswith("_mask.png"):
            continue
        parsed = parse_row(p, root)
        if parsed is not None:
            rows.append(parsed)
            continue
        if args.include_originals:
            leaf_dir = p.parent
            mask_path = find_mask_in_leaf(leaf_dir)
            if mask_path is not None:
                modality_dir = leaf_dir.parent
                health_dir = modality_dir.parent if modality_dir is not None else None
                modality = modality_dir.name if modality_dir is not None else ""
                health = health_dir.name if health_dir is not None else ""
                stem = p.stem
                z_match = ZIDX_RE.search(stem)
                z_index = z_match.group(1) if z_match else ""
                rows.append({
                    "base_id": stem,
                    "modality": modality,
                    "health": health,
                    "z_index": z_index,
                    "distortion": "original",
                    "severity": "0",
                    "path_img": str(p),
                    "path_mask_original": str(mask_path),
                })
    rows.sort(key=lambda r: (r["health"], r["modality"], r["base_id"], r["distortion"], str(r["severity"])))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["base_id", "modality", "health", "z_index", "distortion", "severity", "path_img", "path_mask_original"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {out_csv}")
if __name__ == "__main__":
    main()
