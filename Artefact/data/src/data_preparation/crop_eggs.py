"""
Batch-crops candled egg images to the egg, removing the dark background.

Usage (Windows, in the Egg_Crack_Dataset folder):
    pip install opencv-python numpy
    python crop_eggs.py "01_renamed" "02_cropped"
    python crop_eggs.py "01_renamed" "02_cropped" --mask    (also blacks out background)

Output:
    02_cropped/            cropped square images, same names (E001.jpg ...)
    02_cropped/needs_review/  originals the script could not crop confidently
    02_cropped/crop_log.csv   what happened to every file
The input folder is never modified.
"""
import sys, csv, shutil
from pathlib import Path
import cv2
import numpy as np

PAD = 0.10          # 10% margin around the egg so edge cracks are not cut off
DETECT_W = 800      # detection runs on a small copy for speed

def read(p):  # works with any Windows path
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)

def write(p, img):
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(p))

def find_egg(img):
    """Return (contour, scale) of the egg, or None if nothing egg-like is found."""
    scale = DETECT_W / img.shape[1]
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    total = gray.size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # Try Otsu first, then stricter thresholds (helps when the box walls are lit)
    for t in [otsu] + list(range(int(otsu) + 15, 250, 15)):
        _, m = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        solidity = area / max(cv2.contourArea(cv2.convexHull(c)), 1)
        # Egg-like checks: sensible size, roughly oval, not touching the image border
        if (0.01 < area / total < 0.5 and 0.5 < w / h < 2.0 and solidity > 0.90
                and x > 0 and y > 0 and x + w < m.shape[1] and y + h < m.shape[0]):
            return c, scale
    return None

def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    use_mask = "--mask" in sys.argv
    review = dst / "needs_review"
    dst.mkdir(parents=True, exist_ok=True); review.mkdir(exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    log = []
    for i, p in enumerate(files, 1):
        img = read(p)
        found = find_egg(img) if img is not None else None
        if found is None:
            shutil.copy2(p, review / p.name)
            log.append([p.name, "NEEDS_REVIEW", "", "", "", ""])
            print(f"[{i}/{len(files)}] {p.name}: could not find egg -> needs_review")
            continue
        c, s = found
        c_full = (c / s).astype(np.int32)
        x, y, w, h = cv2.boundingRect(c_full)
        side = int(max(w, h) * (1 + 2 * PAD))           # square crop, no distortion later
        cx, cy = x + w // 2, y + h // 2
        H, W = img.shape[:2]
        x0, y0 = max(cx - side // 2, 0), max(cy - side // 2, 0)
        x1, y1 = min(x0 + side, W), min(y0 + side, H)
        out = img.copy()
        if use_mask:
            mask = np.zeros((H, W), np.uint8)
            cv2.drawContours(mask, [cv2.convexHull(c_full)], -1, 255, -1)
            grow = int(max(w, h) * 0.03)                 # small safety margin around shell
            mask = cv2.dilate(mask, np.ones((grow, grow), np.uint8))
            out[mask == 0] = 0
        write(dst / (p.stem + ".jpg"), out[y0:y1, x0:x1])
        log.append([p.name, "OK", x0, y0, x1 - x0, y1 - y0])
        print(f"[{i}/{len(files)}] {p.name}: OK")
    with open(dst / "crop_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "status", "crop_x", "crop_y", "crop_w", "crop_h"])
        w.writerows(log)
    bad = sum(r[1] != "OK" for r in log)
    print(f"\nDone: {len(log) - bad} cropped, {bad} need manual review.")

if __name__ == "__main__":
    main()
