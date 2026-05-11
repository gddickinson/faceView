"""Build a labeled grid from GUI tour screenshots — crops just the
camera panel out of the full GUI screenshot so the grid is dense."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def crop_camera_panel(img: np.ndarray) -> np.ndarray:
    """Crop to a tight box around the avatar inside the GUI camera
    panel. The GUI is 1280x800 at 1x DPI but macOS Retina screenshots
    return 2560x1600 — scale crop bounds accordingly."""
    h, w = img.shape[:2]
    retina = w > 1500
    if retina:
        # Tight crop around avatar at 2x DPI
        return img[440:1200, 460:980]
    else:
        return img[220:600, 230:490]


def label_image(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img.astype(np.uint8))
    drw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    drw.rectangle((0, 0, 200, 22), fill=(0, 0, 0))
    drw.text((5, 3), text, fill=(255, 235, 180), font=font)
    return np.asarray(pil)


def main():
    gender = sys.argv[1] if len(sys.argv) > 1 else "male"
    tour_dir = Path(f"/tmp/gui_tour_{gender}")
    files = sorted(tour_dir.glob("*.png"))
    if not files:
        print(f"no files in {tour_dir}")
        return
    cells = []
    for p in files:
        img = np.asarray(Image.open(p))
        crop = crop_camera_panel(img)
        labeled = label_image(crop, p.stem)
        cells.append(labeled)
    cols = 6
    rows = (len(cells) + cols - 1) // cols
    blank = np.zeros_like(cells[0])
    while len(cells) < rows * cols:
        cells.append(blank)
    rows_img = [np.hstack(cells[i*cols:(i+1)*cols]) for i in range(rows)]
    grid = np.vstack(rows_img)
    out = f"/tmp/gui_tour_{gender}_grid.png"
    Image.fromarray(grid).save(out)
    print(f"wrote {out} ({len(files)} cells)")


if __name__ == "__main__":
    main()
