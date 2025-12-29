#!/usr/bin/env python3
"""
Create train/val/test file lists by VIDEO folders (prevents leakage).

Assumes frames are stored as:
frames_root/
  M01/
  A01/
  E01/
  ...

Produces:
out_dir/train.txt
out_dir/val.txt
out_dir/test.txt
"""
import argparse
from pathlib import Path

def gather_images(frames_root: Path, video_ids):
    imgs = []
    for vid in video_ids:
        folder = frames_root / vid
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder for video id: {vid} at {folder}")
        imgs.extend(sorted(folder.glob("*.jpg")))
        imgs.extend(sorted(folder.glob("*.png")))
    return sorted(set(imgs))

def write_list(paths, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    out_dir = Path(args.out_dir)

    train_imgs = gather_images(frames_root, args.train)
    val_imgs = gather_images(frames_root, args.val)
    test_imgs = gather_images(frames_root, args.test)

    # Safety: no overlap
    if set(train_imgs) & set(val_imgs):
        raise SystemExit("ERROR: Overlap between train and val images.")
    if set(train_imgs) & set(test_imgs):
        raise SystemExit("ERROR: Overlap between train and test images.")
    if set(val_imgs) & set(test_imgs):
        raise SystemExit("ERROR: Overlap between val and test images.")

    write_list(train_imgs, out_dir / "train.txt")
    write_list(val_imgs, out_dir / "val.txt")
    write_list(test_imgs, out_dir / "test.txt")

    print(f"Train images: {len(train_imgs)}")
    print(f"Val images:   {len(val_imgs)}")
    print(f"Test images:  {len(test_imgs)}")
    print(f"Lists written to: {out_dir}")

if __name__ == "__main__":
    main()
