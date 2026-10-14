"""Run directly to adjust simplepipeline's portrait effect; no files are saved."""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "data/lfw-deepfunneled/lfw-deepfunneled/Bill_Gates/Bill_Gates_0001.jpg"
EDGE_SETTINGS = dict(color_edges=True, gamma=1.0, spread=0, mix=0.0,
                     threshold=50, close_radius=3, close_bottom=True, shape="fill")
BLUR_SIZE = 31
SHARPEN_AMOUNT = 1.0
FEATHER_SIGMA = 0.0


# Same edge and fixed-kernel algorithms as simplepipeline.py, without importing its demo.
def bg_distance(img, border=10):
    """0..1 per pixel: how different its color is from the average color
    along the image border (which is usually background)."""
    f = img.astype(np.float64)
    h, w = f.shape[:2]
    is_border = np.ones((h, w), bool)
    is_border[border:-border, border:-border] = False
    bg_color = f[is_border].mean(axis=0)
    d = np.linalg.norm(f - bg_color, axis=2)
    return d / d.max()


def enhance_edges(img, edges, gamma=1.0, spread=0, mix=0.0):
    """Modify the edge image before thresholding so outlines close better.
    gamma:  transfer function e**gamma; < 1 lifts weak edges (e.g. 0.5)
    spread: Gaussian blur sigma; thickens edges so small gaps join up
    mix:    0..1, how much of the original image to blend in, as each
            pixel's color difference from the background color
    With the defaults the edges are returned unchanged."""
    e = edges.astype(np.float64) / 255                  # 0..1
    e = e ** gamma
    if spread > 0:
        e = cv2.GaussianBlur(e, (0, 0), spread)
        e = e / e.max()
    if mix > 0:
        e = (1 - mix) * e + mix * bg_distance(img)
    return (e * 255).astype(np.uint8)


def make_mask(img, threshold=50, radial_frac=0.5, hull_threshold=50,
              color_edges=False, gamma=1.0, spread=0, mix=0.0, close_radius=0,
              close_bottom=False, shape="hull", return_steps=False):
    """Returns a mask that is 1 on the subject and 0 on the background.
    threshold:    edge threshold 0-255, or None for Otsu (automatic)
    color_edges:  take the strongest Sobel edge over the B, G and R channels,
                  so outlines between colors of similar brightness still show
    close_radius: morphological closing radius; 0 = skip
    close_bottom: treat the bottom image border as an edge, since the
                  shoulders usually run off the bottom of a portrait
    shape:        "hull" = convex hull around strong central edges
                  "fill" = the filled region under the image center
                  (follows the real outline, so edge changes show up)
    return_steps: also return the intermediate images (for inspection)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel edge strength (on gray, or on each color channel)
    channels = cv2.split(img) if color_edges else [gray]
    magnitude = 0
    for ch in channels:
        gx = cv2.Sobel(ch, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(ch, cv2.CV_64F, 0, 1)
        magnitude = np.maximum(magnitude, cv2.magnitude(gx, gy))
    edges = cv2.convertScaleAbs(magnitude)
    edges = enhance_edges(img, edges, gamma, spread, mix)

    # Threshold, close gaps, then fill regions enclosed by edges
    if threshold is None:
        _, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)
    if close_radius > 0:
        size = 2 * close_radius + 1
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, disk)
    if close_bottom:
        binary[-1, :] = 255
    flood = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(flood, None, (0, 0), 255)
    filled = binary | ~flood[1:-1, 1:-1]

    # Weight: 1 at the center, 0 at radial_frac of the way to the corner
    h, w = gray.shape
    Y, X = np.mgrid[:h, :w]
    dist = np.hypot(X - w / 2, Y - h / 2)
    radial = np.clip(1 - dist / (radial_frac * np.hypot(w / 2, h / 2)), 0, 1)

    mask = np.zeros((h, w), np.float32)
    if shape == "fill":
        # The filled region containing the image center, cut off where
        # the radial weight reaches 0
        n, labels = cv2.connectedComponents(np.where(radial > 0, filled, 0).astype(np.uint8))
        center = labels[h // 2, w // 2]
        if center > 0:
            mask[labels == center] = 1.0
    else:
        # Convex hull around the strong edges near the center
        weighted = edges * radial * (filled > 0)
        points = cv2.findNonZero((weighted > hull_threshold).astype(np.uint8))
        if points is not None:
            cv2.fillPoly(mask, [cv2.convexHull(points)], 1.0)

    if return_steps:
        return mask, {"edges": edges, "binary": binary, "filled": filled}
    return mask


def sharpen(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def process_image(image, blur_size=31, sharpen_amount=1.0,
                  feather_sigma=0.0, threshold=50):
    """Original simplepipeline processing, plus optional GUI effect controls."""
    if (not np.isfinite([blur_size, sharpen_amount, feather_sigma]).all()
            or min(blur_size, sharpen_amount, feather_sigma) < 0):
        raise ValueError("Blur, sharpening and feathering must be finite and nonnegative")
    settings = {**EDGE_SETTINGS, "threshold": threshold}
    mask, steps = make_mask(image, return_steps=True, **settings)
    alpha = mask.copy()
    if feather_sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather_sigma)
    alpha = np.clip(alpha, 0, 1)
    sharp = sharpen(image)
    if sharpen_amount != 1.0:
        sharp = cv2.addWeighted(image, 1 - sharpen_amount, sharp, sharpen_amount, 0)
    blur_size = max(1, int(blur_size))
    if blur_size % 2 == 0:
        blur_size += 1
    background = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    a = alpha[..., None]
    result = a * sharp + (1 - a) * background
    steps.update(mask=mask, alpha=alpha)
    return result.astype(np.uint8), alpha, steps


def make_preview(image, result, alpha, steps):
    """Display source, edge-processing stages, mask and final portrait."""
    h, w = image.shape[:2]
    scale = min(360 / w, 360 / h, 1.0)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    panels = []
    for title, picture in (
        ("Original", image), ("Sobel edges", steps["edges"]),
        ("Binary / closed", steps["binary"]), ("Filled", steps["filled"]),
        ("Subject alpha", np.rint(alpha * 255).astype(np.uint8)), ("Portrait", result),
    ):
        if picture.ndim == 2:
            picture = cv2.cvtColor(picture, cv2.COLOR_GRAY2BGR)
        panel = cv2.resize(picture, size, interpolation=cv2.INTER_AREA)
        panel = cv2.copyMakeBorder(panel, 28, 0, 0, 0, cv2.BORDER_CONSTANT,
                                   value=(35, 35, 35))
        text_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        font_scale = 0.45 * min(1.0, max(1, size[0] - 12) / max(1, text_width))
        cv2.putText(panel, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (240, 240, 240), 1, cv2.LINE_AA)
        panels.append(panel)
    return np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))


def run_GUI(image):
    window = "Simple Pipeline"
    controls = {
        "BG blur": (round(BLUR_SIZE), 101),
        "Sharpen": (round(SHARPEN_AMOUNT * 10), 30),
        "Feather*10": (round(FEATHER_SIGMA * 10), 200),
        "Threshold": (EDGE_SETTINGS.get("threshold", 50) or 0, 255),
    }
    ready = False
    current = None

    def on_trackbar(_):
        nonlocal current
        if not ready:
            return
        values = {name: cv2.getTrackbarPos(name, window) for name in controls}
        current = process_image(
            image, blur_size=values["BG blur"],
            sharpen_amount=values["Sharpen"] / 10,
            feather_sigma=values["Feather*10"] / 10,
            threshold=values["Threshold"] or None)
        result, alpha, steps = current
        cv2.imshow(window, make_preview(image, result, alpha, steps))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(window, 1000, 800)
        for name, (value, maximum) in controls.items():
            cv2.createTrackbar(name, window, max(0, min(value, maximum)), maximum, on_trackbar)
        ready = True
        on_trackbar(0)
        print("Sharpen / Feather values are divided by 10; Threshold 0 selects Otsu.")
        print("Q / Esc: close without saving.")
        while True:
            if (cv2.waitKey(30) & 0xFF) in (27, ord("q")):
                break
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        cv2.destroyAllWindows()
    return current


def main():
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")
    return run_GUI(image)


if __name__ == "__main__":
    main()
