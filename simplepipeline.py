import cv2
import numpy as np


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


def fourier_mask(img, cutoff=20, smooth=8, radial_frac=0.5):
    """Mask from the Fourier domain: keep only high frequencies (fine
    detail), measure how much detail each area has, and take the
    detailed area near the center as the subject.
    cutoff: radius of the low frequencies removed (bigger = only finer detail)
    smooth: how widely detail is spread out when measuring it"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    Y, X = np.mgrid[:h, :w]

    # High-pass filter: 0 at the center of the shifted spectrum, 1 far away
    F = np.fft.fftshift(np.fft.fft2(gray))
    highpass = 1 - np.exp(-((X - w / 2) ** 2 + (Y - h / 2) ** 2) / (2 * cutoff ** 2))
    detail = np.real(np.fft.ifft2(np.fft.ifftshift(F * highpass)))

    # Local detail energy, weighted toward the image center
    energy = cv2.GaussianBlur(np.abs(detail), (0, 0), smooth)
    dist = np.hypot(X - w / 2, Y - h / 2)
    radial = np.clip(1 - dist / (radial_frac * np.hypot(w / 2, h / 2)), 0, 1)
    score = cv2.normalize(energy * radial, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu picks the threshold; convex hull makes one solid shape
    _, binary = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    points = cv2.findNonZero(binary)
    mask = np.zeros((h, w), np.float32)
    if points is not None:
        cv2.fillPoly(mask, [cv2.convexHull(points)], 1.0)
    return mask


def sharpen(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def portrait(img, method="edges", blur_size=31, **edge_settings):
    """Sharp subject over a blurred background. blur_size must be odd.
    method: "edges" (Sobel) or "fourier" (high-pass detail energy)
    edge_settings: passed to make_mask, e.g. gamma=0.5, threshold=None"""
    if method == "fourier":
        mask = fourier_mask(img)
    else:
        mask = make_mask(img, **edge_settings)
    mask = mask[..., None]                    # (h, w) -> (h, w, 1)
    sharp = sharpen(img)
    blurred = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    result = mask * sharp + (1 - mask) * blurred
    return result.astype(np.uint8), mask


img = cv2.imread("./data/lfw-deepfunneled/lfw-deepfunneled/Bill_Gates/Bill_Gates_0001.jpg")

# Edge settings to experiment with (all defaults = original behavior)
settings = dict(color_edges=True, gamma=1.0, spread=0, mix=0.0,
                threshold=50, close_radius=3, close_bottom=True, shape="fill")

mask, steps = make_mask(img, return_steps=True, **settings)
result, _ = portrait(img, **settings)

cv2.imshow("image", img)
cv2.imshow("enhanced edges", steps["edges"])
cv2.imshow("binary (after closing)", steps["binary"])
cv2.imshow("filled", steps["filled"])
cv2.imshow("mask", mask)
cv2.imshow("portrait", result)
cv2.waitKey(0)
cv2.destroyAllWindows()