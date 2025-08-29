import sys
import argparse
from pathlib import Path
from PIL import Image
def fix_png(path: Path):
    """
    Open the PNG at `path`, remove any ICC profile, re-save, and replace.
    """
    try:
        with Image.open(path) as img:
            img_info = img.info
            if "icc_profile" in img_info:
                img_info.pop("icc_profile")
            mode = "RGBA" if img.mode == "RGBA" else "RGB"
            clean = img.convert(mode)
            temp_path = path.with_suffix(".tmp.png")
            clean.save(temp_path, format="PNG")
        temp_path.replace(path)
        print(f"[OK]   {path}")
    except Exception as e:
        print(f"[FAIL] {path}  {e}")
def main(root_dir: Path):
    if not root_dir.is_dir():
        print(f"Error: {root_dir} is not a directory.")
        sys.exit(1)
    pngs = list(root_dir.rglob("*.png"))
    print(f"Found {len(pngs)} PNG files under {root_dir}\n")
    for p in pngs:
        fix_png(p)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip incorrect sRGB ICC profiles from PNG files.")
    parser.add_argument("--dataset_dir", type=str, help="Root of the dataset to process.")
    args = parser.parse_args()
    main(Path(args.dataset_dir))
