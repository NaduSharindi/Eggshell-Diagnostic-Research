"""
Semi-automatic cropping for the images in needs_review.
For each image it shows a PROPOSED crop box (green). You decide with one key:

    Enter / Space : accept the green box
    d             : draw your own box with the mouse, then press Enter
    p             : accept, but mark as PARTIAL (part of the egg is outside the photo)
    x             : exclude this image (unusable), nothing is saved
    b             : go back one image
    q             : quit (progress is saved; run again to continue)

Usage (needs the normal opencv-python, not the headless version):
    python review_crop.py "02_cropped/needs_review" "02_cropped"
"""
import sys, csv
from pathlib import Path
import cv2
import numpy as np

PAD = 0.10
SCREEN_H = 850   # display height; lower it if the window is too big for your screen

def read(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)

def write(p, img):
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(p))

def propose(img):
    """Relaxed egg detection: allows eggs touching the edge or filling the frame."""
    s = 800 / img.shape[1]
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for t in [otsu] + list(range(int(otsu) + 15, 250, 15)):
        _, m = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        a = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        sol = a / max(cv2.contourArea(cv2.convexHull(c)), 1)
        if 0.01 < a / gray.size < 0.95 and 0.4 < w / h < 2.5 and sol > 0.8:
            return tuple(int(v / s) for v in (x, y, w, h))
    H, W = img.shape[:2]                      # fallback: centre square
    side = min(H, W)
    return ((W - side) // 2, (H - side) // 2, side, side)

def square_crop(img, box):
    """Square crop with margin; pads with black if the square runs off the photo."""
    x, y, w, h = box
    side = int(max(w, h) * (1 + 2 * PAD))
    x0, y0 = x + w // 2 - side // 2, y + h // 2 - side // 2
    H, W = img.shape[:2]
    top, left = max(-y0, 0), max(-x0, 0)
    bottom, right = max(y0 + side - H, 0), max(x0 + side - W, 0)
    crop = img[max(y0, 0):min(y0 + side, H), max(x0, 0):min(x0 + side, W)]
    return cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    log_path = dst / "review_log.csv"
    done = {}
    if log_path.exists():
        with open(log_path, newline="") as f:
            done = {r["file"]: r for r in csv.DictReader(f)}
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    todo = [p for p in files if p.name not in done]
    print(f"{len(files)} images, {len(todo)} left to review.")
    win = "Review crop"
    i = 0
    while 0 <= i < len(todo):
        p = todo[i]
        img = read(p)
        box = propose(img)
        sc = SCREEN_H / img.shape[0]
        view = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        x, y, w, h = (int(v * sc) for v in box)
        show = view.copy()
        cv2.rectangle(show, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(show, f"{p.name} ({i+1}/{len(todo)})  Enter=ok d=draw p=partial x=exclude b=back q=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(win, show)
        key = cv2.waitKey(0) & 0xFF
        status = None
        if key in (13, 32):
            status = "OK"
        elif key == ord("p"):
            status = "PARTIAL"
        elif key == ord("d"):
            r = cv2.selectROI(win, view, showCrosshair=True)
            if r[2] > 0 and r[3] > 0:
                box = tuple(int(v / sc) for v in r)
                status = "OK_MANUAL"
        elif key == ord("x"):
            status = "EXCLUDED"
        elif key == ord("b"):
            i = max(i - 1, 0)
            done.pop(todo[i].name, None)
            continue
        elif key == ord("q"):
            break
        if status is None:
            continue
        if status != "EXCLUDED":
            write(dst / (p.stem + ".jpg"), square_crop(img, box))
        done[p.name] = {"file": p.name, "status": status, "box_x": box[0], "box_y": box[1],
                        "box_w": box[2], "box_h": box[3]}
        with open(log_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["file", "status", "box_x", "box_y", "box_w", "box_h"])
            wr.writeheader(); wr.writerows(done.values())
        i += 1
    cv2.destroyAllWindows()
    print(f"Reviewed {len(done)} of {len(files)}. Log: {log_path}")

if __name__ == "__main__":
    main()
