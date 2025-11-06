import argparse
import math
from pathlib import Path
import cv2
import os

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def get_video_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        return 30.0
    return float(fps)

def iter_videos(video_root: Path):
    """Yield (video_path, rel_dir) where rel_dir is the subfolder under video_root."""
    for p in video_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            rel_dir = p.parent.relative_to(video_root)  
            yield p, rel_dir

def extract_from_video(vp: Path, rel_dir: Path, out_root: Path,
                       target_fps: float, start_sec: float, end_sec: float):
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {vp}")
        return 0

    orig_fps = get_video_fps(cap)
    frame_step = max(1, int(round(orig_fps / max(target_fps, 0.01))))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    start_frame = int(start_sec * orig_fps)
    end_frame = int(end_sec * orig_fps) if end_sec > 0 else total_frames

    
    out_dir = out_root / rel_dir
    ensure_dir(out_dir)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        i = start_frame
    else:
        i = 0

    saved = 0
    base = vp.stem  

    print(f"[INFO] {vp.name}: orig_fps={orig_fps:.2f}, step={frame_step}, frames={total_frames}, out={out_dir}")
    while True:
        if end_frame and i >= end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if (i - start_frame) % frame_step == 0:
            outp = out_dir / f"{base}_f{i:06d}.jpg"
            cv2.imwrite(str(outp), frame)
            saved += 1
        i += 1

    cap.release()
    return saved

def main():
    ap = argparse.ArgumentParser(description="Convert videos to images (frame extraction)")
    ap.add_argument("--video_root", type=Path, default=Path("data/videos"),
                    help="Root folder containing videos (e.g., data/videos/video1/*.mp4)")
    ap.add_argument("--out_root", type=Path, default=Path("data/frames_raw"),
                    help="Output root for extracted frames")
    ap.add_argument("--fps", type=float, default=2.0,
                    help="Target extracted FPS (e.g., 1–3)")
    ap.add_argument("--start_sec", type=float, default=0.0,
                    help="Start time in seconds (0 = from beginning)")
    ap.add_argument("--end_sec", type=float, default=0.0,
                    help="End time in seconds (0 = until end)")
    args = ap.parse_args()

    ensure_dir(args.out_root)

    total_saved = 0
    total_vids = 0

    for vp, rel_dir in iter_videos(args.video_root):
        saved = extract_from_video(
            vp=vp,
            rel_dir=rel_dir,
            out_root=args.out_root,
            target_fps=args.fps,
            start_sec=args.start_sec,
            end_sec=args.end_sec
        )
        print(f"[DONE] {vp.name}: saved {saved} frames")
        total_saved += saved
        total_vids += 1

    if total_vids == 0:
        print(f"[INFO] No videos found under: {args.video_root} (looked for {sorted(VIDEO_EXTS)})")
    else:
        print(f"[SUMMARY] Videos: {total_vids}, Frames saved: {total_saved}, Output: {args.out_root}")

if __name__ == "__main__":
    main()