#!/usr/bin/env python3
"""
Stratified frame sampling for traffic videos.

Default: sample 20 frames/video (roughly early/mid/late thirds).
Why: reduce redundancy, preserve temporal diversity, keep annotation workload manageable.

Usage:
python sampling/sample_stratified.py --video_dir <dir> --out_dir <dir> --per_video 20
"""
import argparse
from pathlib import Path
import cv2

def linspace_times(start: float, end: float, n: int):
    if n <= 0:
        return []
    if n == 1:
        return [(start + end) / 2.0]
    step = (end - start) / (n + 1)
    return [start + step * (i + 1) for i in range(n)]

def distribute_counts(per_video: int):
    base = per_video // 3
    rem = per_video % 3
    counts = [base, base, base]
    # symmetric distribution: prefer early & late
    if rem == 1:
        counts[1] += 1
    elif rem == 2:
        counts[0] += 1
        counts[2] += 1
    return counts  # early, mid, late

def sample_video(video_path: Path, out_dir: Path, per_video: int = 20):
    counts = distribute_counts(per_video)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Bad metadata for {video_path}. fps={fps}, frames={frame_count}")

    duration_sec = frame_count / fps

    segments = [
        (0.0, duration_sec / 3.0),
        (duration_sec / 3.0, 2.0 * duration_sec / 3.0),
        (2.0 * duration_sec / 3.0, duration_sec),
    ]

    times = []
    for (s, e), n in zip(segments, counts):
        times.extend(linspace_times(s, e, n))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    saved = 0
    for idx, t in enumerate(times, start=1):
        frame_idx = int(t * fps)
        frame_idx = max(0, min(frame_idx, frame_count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = out_dir / f"{stem}_S{idx:02d}_t{int(t):06d}s.jpg"
        if cv2.imwrite(str(fname), frame):
            saved += 1

    cap.release()
    return saved, duration_sec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", type=str, required=True, help="Folder containing videos (.mp4 by default)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder for per-video sampled frames")
    ap.add_argument("--per_video", type=int, default=20, help="Frames sampled per video (default: 20)")
    ap.add_argument("--ext", type=str, default="mp4", help="Video file extension (default: mp4)")
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    out_root = Path(args.out_dir)
    videos = sorted(video_dir.glob(f"*.{args.ext}"))

    if not videos:
        raise SystemExit(f"No videos found in {video_dir} with extension .{args.ext}")

    total = 0
    for v in videos:
        out_dir = out_root / v.stem
        saved, dur = sample_video(v, out_dir, per_video=args.per_video)
        total += saved
        print(f"[OK] {v.name}: duration={dur/60:.1f} min, sampled={saved} -> {out_dir}")

    print(f"\nDone. Total sampled frames: {total}")

if __name__ == "__main__":
    main()
