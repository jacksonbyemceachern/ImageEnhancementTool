"""
portrait_mask.py -- build a subject mask from edges, with swappable steps.

Pipeline:
    gray -> edges -> binarize -> close gaps -> fill holes -> radial crop
         -> mask by "component" (center blob) or "hull" (convex hull of edges)
         -> optional feathering

Edit the settings at the bottom of the file, then run:
    python portrait_mask.py
"""
from pathlib import Path
import cv2
import numpy as np


# ---------------------------------------------------------------- parameters

#Sharpening Parameters - SIGMA, tp_power, sharp Kernel, (Median Filtering), blur kernel size

DEFAULTS = {
    "blur_sigma": 0,        # pre-blur; 1.1 == GaussianBlur((5,5), 0)
    "edges": "sobel",         # "sobel" or "canny"
    "sobel_ksize": 3,
    "canny_low": 50,
    "canny_high": 150,
    "threshold": 50,          # on 0-255 edge strength; None -> Otsu
    "close_radius": 3,        # closing disk radius; 0 = skip
    "radial_frac": 0.5,       # radial weight reaches 0 at this fraction of
                              # the center-to-corner distance
    "method": "hull",    # "component" or "hull"
    "connectivity": 8,        # 4 or 8 (component method)
    "hull_threshold": 50,     # min radially weighted edge strength (hull)
    "feather_sigma": 0,       # soften mask edge; 0 = hard mask
}

# Each preset only lists what differs from DEFAULTS.
PRESETS = {
    "sobel_component": {"method": "component"},
    "canny_component": {"edges": "canny", "method": "component"},
    "sobel_hull":      {"method": "hull"},
    "canny_hull":      {"edges": "canny", "method": "hull"},
    "sobel_otsu_wide": {"threshold": None, "radial_frac": 0.7},
}


def get_params(preset="sobel_component", **overrides):
    """Defaults, then the preset's changes, then any overrides you pass.
    Example: get_params("sobel_hull", close_radius=5)"""
    p = dict(DEFAULTS)
    p.update(PRESETS[preset])
    p.update(overrides)
    return p


# ------------------------------------------------------------ pipeline steps

def disk(radius):
    """Circular structuring element of the given radius."""
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def to_gray(img_bgr, blur_sigma):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    return gray


def edge_strength(gray, p):
    """Edge map as uint8 0-255. Canny gives 0/255 directly; Sobel gradient
    magnitude is rescaled so the strongest edge in the image is 255."""
    if p["edges"] == "canny":
        return cv2.Canny(gray, p["canny_low"], p["canny_high"])
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=p["sobel_ksize"])
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=p["sobel_ksize"])
    mag = cv2.magnitude(gx, gy)
    return cv2.convertScaleAbs(mag)


def binarize(edges, threshold):
    if threshold is None:
        _, b = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, b = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)
    return b


def close_gaps(binary, radius):
    """Closing = dilate then erode: bridges small breaks in outlines."""
    if radius <= 0:
        return binary
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, disk(radius))


def fill_holes(binary):
    """Flood the background from outside; whatever the flood could not
    reach is enclosed by edges (a hole), so add it to the mask."""
    h, w = binary.shape
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)   # floodFill needs +2 px
    cv2.floodFill(padded, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(padded)[1:-1, 1:-1]
    return cv2.bitwise_or(binary, holes)


def radial_weight(shape, frac):
    """1 at the image center, falling linearly to 0 at `frac` of the
    center-to-corner distance."""
    if frac <= 0:
        raise ValueError("radial_frac must be positive")
    h, w = shape
    Y, X = np.mgrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    return np.clip(1 - dist / (frac * max_dist), 0, 1).astype(np.float32)


def center_component(binary, connectivity):
    """Keep the connected component under the image center."""
    n, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
    h, w = binary.shape
    center_label = labels[h // 2, w // 2]
    if center_label == 0:
        print("warning: image center is background; mask is empty")
        return labels, np.zeros_like(binary)
    return labels, np.where(labels == center_label, 255, 0).astype(np.uint8)


def hull_mask(weighted, threshold):
    """Convex hull around all pixels whose weighted edge strength
    exceeds the threshold."""
    strong = np.where(weighted > threshold, 255, 0).astype(np.uint8)
    out = np.zeros_like(strong)
    contours, _ = cv2.findContours(strong, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("warning: no edges above hull threshold; mask is empty")
        return strong, out
    hull = cv2.convexHull(np.vstack(contours))
    cv2.fillPoly(out, [hull], 255)
    return strong, out


def feather(mask, sigma):
    """Hard 0/255 mask -> soft alpha in [0, 1]."""
    alpha = mask.astype(np.float32) / 255
    if sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
    return np.clip(alpha, 0.0, 1.0)


# --------------------------------------------------------------- full recipe

def make_mask(img_bgr, p):
    """Run the pipeline. Returns (alpha in [0,1], dict of every step)."""
    s = {}
    s["gray"] = to_gray(img_bgr, p["blur_sigma"])
    s["edges"] = edge_strength(s["gray"], p)
    s["binary"] = binarize(s["edges"], p["threshold"])
    s["closed"] = close_gaps(s["binary"], p["close_radius"])
    s["filled"] = fill_holes(s["closed"])
    s["radial"] = radial_weight(s["binary"].shape, p["radial_frac"])
    s["cropped"] = np.where(s["radial"] > 0, s["filled"], 0).astype(np.uint8)

    if p["method"] == "component":
        s["labels"], s["mask"] = center_component(s["cropped"], p["connectivity"])
    else:  # "hull"
        inside = s["cropped"] > 0
        s["weighted"] = s["edges"].astype(np.float32) * s["radial"] * inside
        s["strong"], s["mask"] = hull_mask(s["weighted"], p["hull_threshold"])

    s["alpha"] = feather(s["mask"], p["feather_sigma"])
    return s["alpha"], s


def mask_and_complement(img_bgr, preset="sobel_component", **overrides):
    """Main entry point. Returns (mask, complement), both float in [0, 1]:
    mask is 1 on the subject, complement is 1 on the background.
    Example: fg, bg = mask_and_complement(img, "sobel_hull", close_radius=5)"""
    alpha, _ = make_mask(img_bgr, get_params(preset, **overrides))
    return alpha, 1.0 - alpha

# ------------------------------------------------------------------ Sharpening

def inv_tophat(img_size, sigma=20):
    #Ranges from 0 in origo (125,125) to 1 in an inverse super-gaussian profile
    tp_power = 1

    y, x = np.ogrid[:img_size, :img_size]
    center = img_size // 2
    dist = (x - center)**2 + (y - center)**2
    gauss = np.exp(-(dist / (2 * sigma**2))**tp_power)
    inv_tophat = 1-gauss
    return inv_tophat

def hp_filtering(gray, gray_part, sigma_num, sobel_poly):
    fourier_part = np.fft.fft2(gray)
    fourier_coeff = np.fft.fftshift(fourier_part)

    spectrum1 = np.log(1 + np.abs(fourier_coeff))
    spectrum1 = cv2.normalize(spectrum1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("Spectrum", spectrum1)

    img_size = gray.shape[0]
    tp = inv_tophat(img_size, sigma=sigma_num)#sigma = 50, 80)
    hp_freqs = np.multiply(tp,fourier_coeff)

    spectrum2 = np.log(1 + np.abs(hp_freqs))
    spectrum2 = cv2.normalize(spectrum2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("Spectrum2", spectrum2)

    hp = np.real(np.fft.ifft2(np.fft.ifftshift(hp_freqs)))
    #print(hp.min(), hp.max(),hp)
    hp_norm = cv2.normalize(hp,None, 0,255,cv2.NORM_MINMAX)#.astype(np.uint8)
    #print(hp_norm.min(), hp_norm.max(),hp_norm)
    #cv2.imshow("hp2",hp_norm.astype(np.uint8))

    #Get face part of image 
    hp_part = np.multiply(sobel_poly//255,hp) #HPNORM OR HP?????????
    #cv2.imshow("hp_part", hp_part.astype(np.uint8))

    gray_float = gray_part.astype(np.float64)
    enhanced = gray_float+hp_part
    cv2.imshow("hp_enhanced", enhanced.astype(np.uint8))
    return enhanced

def sharpen(img_part, amount=1.0, radius=1.0):
    """Unsharp masking: image + amount * (image - Gaussian(image))."""
    if amount < 0 or radius <= 0:
        raise ValueError("amount must be nonnegative and radius must be positive")
    if amount == 0:
        return img_part.copy()
    smooth = cv2.GaussianBlur(img_part, (0, 0), radius)
    return cv2.addWeighted(img_part, 1.0 + amount, smooth, -amount, 0)


# def sharpen(img_part, amount=1.0, radius=1.0):
#     """Unsharp masking: image + amount * (image - Gaussian(image))."""
#     if amount < 0 or radius <= 0:
#         raise ValueError("amount must be nonnegative and radius must be positive")
#     if amount == 0:
#         return img_part.copy()
#     smooth = cv2.GaussianBlur(img_part, (0, 0), radius)
#     return cv2.addWeighted(img_part, 1.0 + amount, smooth, -amount, 0)


# ------------------------------------------------------------------- blur
def _as_alpha(mask, shape):
    """Accept uint8 masks in [0,255] or floating-point alpha in [0,1]."""
    mask = np.asarray(mask)
    if mask.shape != shape:
        raise ValueError("mask must have the same height and width as the image")
    alpha = mask.astype(np.float32)
    if mask.dtype == np.uint8:
        alpha /= 255.0
    if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError("use a uint8 0-255 mask or an alpha mask in [0,1]")
    return alpha[..., None]


def _blur_background(image, kernel_size):
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_bokeh_effect(original_image, mask, blur_kernel_size: int = 31):
    """Preserve the foreground and Gaussian-blur the background."""
    return portrait_enhance(original_image, mask, blur_kernel_size, sharpen_amount=0)


def portrait_enhance(original_image, mask, blur_kernel_size=31,
                     sharpen_amount=1.0, equalize_subject=False):
    """Blend enhanced subject and blurred background once using soft alpha."""
    alpha = _as_alpha(mask, original_image.shape[:2])
    subject = original_image.copy()
    if equalize_subject:
        lab = cv2.cvtColor(subject, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        subject = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Enhance the full image before masking to avoid a black mask-edge halo.
    subject = sharpen(subject, amount=sharpen_amount)
    background = _blur_background(original_image, blur_kernel_size)
    result = subject.astype(np.float32) * alpha + background.astype(np.float32) * (1 - alpha)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)

# ------------------------------------------------------------------- display

def to_display(im):
    """Convert any step output to uint8 so cv2.imshow/imwrite show it right."""
    if im.dtype == np.uint8:
        return im
    if np.issubdtype(im.dtype, np.integer):          # label image -> colors
        if im.max() == 0:
            return np.zeros(im.shape, np.uint8)
        col = cv2.applyColorMap((im * 255 // im.max()).astype(np.uint8),
                                cv2.COLORMAP_JET)
        col[im == 0] = 0
        return col
    im = im.astype(np.float32)
    if im.max() > 1:
        im = im / im.max()
    return (np.clip(im, 0, 1) * 255).astype(np.uint8)


def overlay(img_bgr, alpha, dim=0.3):
    """Subject at full brightness, background dimmed."""
    a = alpha[..., None]
    return (img_bgr * (a + (1 - a) * dim)).astype(np.uint8)


def show(images, save_dir=None):
    """Show images in windows, or write them to save_dir as PNGs."""
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        for name, im in images.items():
            cv2.imwrite(str(Path(save_dir) / f"{name}.png"), to_display(im))
        print(f"saved {len(images)} images to {save_dir}")
        return
    for name, im in images.items():
        cv2.imshow(name, to_display(im))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compare_presets(img_bgr):
    """One overlay per preset, tiled horizontally with labels."""
    tiles = []
    for name in PRESETS:
        alpha, _ = make_mask(img_bgr, get_params(name))
        tile = overlay(img_bgr, alpha)
        cv2.putText(tile, name, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    return cv2.hconcat(tiles)


# ---------------------------------------------------------------------- main

if __name__ == "__main__":
    # ---- settings: edit these ----
    IMAGE_PATH = "./data/lfw-deepfunneled/lfw-deepfunneled/Bill_Gates/Bill_Gates_0001.jpg"
    PRESET = "sobel_component"
    OVERRIDES = {}            # e.g. {"close_radius": 5, "connectivity": 4}
    SAVE_DIR = None           # e.g. "results/" to save PNGs instead of showing
    SHOW_STEPS = False        # True: also show every intermediate step
    COMPARE = False           # True: show all presets side by side instead
    # ------------------------------

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"could not read {IMAGE_PATH}")
    params = get_params(PRESET, **OVERRIDES)
    mask, steps = make_mask(img, params)

    im_mask, im_comp = mask_and_complement(img)

    
    cv2.imshow("Steps Gray", steps["gray"])

    
    gray = np.asarray(steps["gray"])
    comp = np.asarray(im_comp)

    gray_part = np.multiply(gray, comp)

    cv2.imshow("gray part", gray_part)

    sharp = hp_filtering(steps["gray"], gray_part=im_mask, sigma_num=20, sobel_poly=mask) 

    cv2.imshow("gray",steps["gray"])

    cv2.imshow("image Mask", im_mask)
    cv2.imshow("sharpened", sharp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
