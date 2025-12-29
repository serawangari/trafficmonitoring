#!/usr/bin/env python3
"""
Prepare Ultralytics dataset folders (images/labels for train/val/test).

After you export annotations in YOLO format, you typically have:
- images/ (jpg)
- labels/ (txt with same basename)

This script copies files into:
dataset_out/
  images/train|val|test
  labels/train|val|test

It uses your split lists (train.txt/val.txt/test.txt) and matches by image basename.
"""
import argparse
from pathlib import Path
import shutil

def load_basenames(list_path: Path):
    with list_path.open("r", encoding="utf-8") as f:
        return [Path(line.strip()).name for line in f if line.strip()]

def copy_split(split_name: str, basenames, imgs_dir: Path, labels_dir: Path, out_root: Path):
    out_img_dir = out_root / "images" / split_name
    out_lbl_dir = out_root / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    copied, missing = 0, 0
    for img_name in basenames:
        src_img = imgs_dir / img_name
        src_lbl = labels_dir / (Path(img_name).stem + ".txt")
        if not src_img.exists() or not src_lbl.exists():
            missing += 1
            continue
        shutil.copy2(src_img, out_img_dir / img_name)
        shutil.copy2(src_lbl, out_lbl_dir / (Path(img_name).stem + ".txt"))
        copied += 1
    return copied, missing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations_images_dir", required=True)
    ap.add_argument("--annotations_labels_dir", required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--dataset_out", required=True)
    args = ap.parse_args()

    imgs_dir = Path(args.annotations_images_dir)
    labels_dir = Path(args.annotations_labels_dir)
    splits_dir = Path(args.splits_dir)
    out_root = Path(args.dataset_out)

    for split in ["train", "val", "test"]:
        basenames = load_basenames(splits_dir / f"{split}.txt")
        copied, missing = copy_split(split, basenames, imgs_dir, labels_dir, out_root)
        print(f"{split}: copied={copied}, missing={missing}")

    # write a dataset yaml next to output for convenience
    yaml_path = out_root / "data.yaml"
    if not yaml_path.exists():
        yaml_path.write_text(
            "path: " + str(out_root) + "\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n\n"
            "names:\n"
            "  0: car\n"
            "  1: bus\n"
            "  2: truck\n"
            "  3: motorcycle\n",
            encoding="utf-8"
        )
        print(f"Wrote {yaml_path} (edit class names if needed)")

    print(f"Dataset ready at: {out_root}")

if __name__ == "__main__":
    main()
