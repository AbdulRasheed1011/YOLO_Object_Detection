import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
import re

from PIL import Image
import imagehash

# Optional SSIM
try:
    import cv2
    from skimage.metrics import structural_similarity as ssim
    _HAS_SSIM = True
except Exception:
    _HAS_SSIM = False


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def natural_key(path: Path) -> Tuple:
    """Sort helper so frames order by name nicely (f2 < f10)."""
    return tuple(int(t) if t.isdigit() else t for t in re.split(r'(\d+)', path.name))


def list_images(root: Path) -> List[Path]:
    imgs: List[Path] = []
    for ext in IMG_EXTS:
        imgs.extend(root.rglob(f"*{ext}"))
    return sorted(imgs, key=natural_key)


def compute_phash(img_path: Path):
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        return imagehash.phash(im)  


def compute_ssim(a_path: Path, b_path: Path) -> float:
    """Grayscale SSIM on overlapping crop (handles slight size differences)."""
    if not _HAS_SSIM:
        return 0.0
    a = cv2.imread(str(a_path), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(b_path), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return 0.0
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if h < 8 or w < 8:
        return 0.0
    return float(ssim(a[:h, :w], b[:h, :w]))


def group_by_subfolder(src_root: Path, imgs: List[Path]):
    """Group images by immediate subfolder relative to src_root."""
    groups = {}
    for p in imgs:
        rel_dir = p.parent.relative_to(src_root)  
        groups.setdefault(rel_dir, []).append(p)
    for k in groups:
        groups[k].sort(key=natural_key)
    return groups


def dedup_group(
    src_root: Path,
    dst_root: Path,
    frames: List[Path],
    phash_thresh: int,
    use_ssim: bool,
    ssim_thresh: float
) -> Tuple[int, int]:
    """
    Deduplicate a single group of frames (one subfolder) sequentially:
      - compare each frame to the last KEPT frame
      - if pHash distance <= threshold => duplicate (skip), unless SSIM < thresh (keep)
    Returns: (kept_count, total_count)
    """
    if not frames:
        return (0, 0)

    kept = 0
    last_kept_img = None
    last_kept_hash = None

    # Prepare destination folder
    rel_dir = frames[0].parent.relative_to(src_root)
    dst_dir = dst_root / rel_dir
    ensure_dir(dst_dir)

    for idx, ip in enumerate(frames):
        try:
            h = compute_phash(ip)
        except Exception as e:
            print(f"[dedup] WARN: cannot hash {ip}: {e}")
            continue

        is_dup = False
        if last_kept_hash is not None:
            dist = h - last_kept_hash  
            if dist <= phash_thresh:
                is_dup = True
               
                if use_ssim and _HAS_SSIM and last_kept_img is not None:
                    s = compute_ssim(last_kept_img, ip)
                    if s < ssim_thresh:  
                        is_dup = False

        if not is_dup:
            shutil.copy2(ip, dst_dir / ip.name)
            last_kept_hash = h
            last_kept_img = ip
            kept += 1
        

    return (kept, len(frames))


def main():
    ap = argparse.ArgumentParser(description="Remove near-duplicate frames (post-dedup)")
    ap.add_argument("--src", type=Path, required=True, help="Source frames root (pre-dedup), e.g., data/frames_raw")
    ap.add_argument("--dst", type=Path, required=True, help="Destination for curated frames, e.g., data/frames")
    ap.add_argument("--phash_thresh", type=int, default=6, help="0–64. Lower => stricter dedup (keep fewer)")
    ap.add_argument("--use_ssim", action="store_true", help="Enable SSIM rescue (requires scikit-image)")
    ap.add_argument("--ssim_thresh", type=float, default=0.95, help="Higher => more aggressive duplicate removal")
    args = ap.parse_args()

    ensure_dir(args.dst)

    imgs = list_images(args.src)
    if not imgs:
        print(f"[dedup] No images found under: {args.src}")
        return

    groups = group_by_subfolder(args.src, imgs)
    total_kept = 0
    total_seen = 0

    print(f"[dedup] Starting post-dedup on {len(imgs)} images, groups={len(groups)}, "
          f"phash_thresh={args.phash_thresh}, use_ssim={args.use_ssim}")
    if args.use_ssim and not _HAS_SSIM:
        print("[dedup] NOTE: scikit-image / OpenCV not available; --use_ssim will be ignored.")

    for rel_dir, frames in groups.items():
        kept, seen = dedup_group(
            src_root=args.src,
            dst_root=args.dst,
            frames=frames,
            phash_thresh=args.phash_thresh,
            use_ssim=args.use_ssim,
            ssim_thresh=args.ssim_thresh
        )
        total_kept += kept
        total_seen += seen
        print(f"[dedup] {rel_dir if str(rel_dir) != '.' else '/'}  kept {kept}/{seen}")

    print(f"[dedup] DONE. Kept {total_kept}/{total_seen} images → {args.dst}")


if __name__ == "__main__":
    main()