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
    "method": "component",    # "component" or "hull"
                              # (FIX: was "hull", so the *_component presets
                              #  silently ran the hull method)
    "connectivity": 8,        # 4 or 8 (component method)
    "hull_threshold": 50,     # min radially weighted edge strength (hull)
    "feather_sigma": 10,       # soften mask edge; 0 = hard mask
}

# Each preset only lists what differs from DEFAULTS.
PRESETS = {
    "sobel_component": {},
    "canny_component": {"edges": "canny"},
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
    magnitude is converted to uint8, so values above 255 saturate at 255
    (FIX: docstring previously claimed it was rescaled to the max)."""
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
    return alpha


# --------------------------------------------------------------- full recipe

def make_mask(img_bgr, p):
    """Run the pipeline. Returns (alpha in [0,1], dict of every step)."""
    s = {}
    s["gray"] = to_gray(img_bgr, p["blur_sigma"])
    s["edges"] = edge_strength(s["gray"], p)
    s["binary"] = binarize(s["edges"], p["threshold"])
    s["closed"] = close_gaps(s["binary"], p["close_radius"])
    s["filled"] = fill_holes(s["closed"])   # FIX: was s["binary"], so closing had no effect
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

def _mask01(mask):
    """Accept a 0-1 alpha or a 0-255 mask; return float 0-1."""
    mask = mask.astype(np.float32)
    return mask / 255.0 if mask.max() > 1 else mask


def inv_tophat(shape, sigma=20):
    #Ranges from 0 in origo (125,125) to 1 in an inverse super-gaussian profile
    # FIX: takes the (h, w) shape so non-square images work; an int still
    # means a square image of that size.
    tp_power = 1

    if np.isscalar(shape):
        shape = (shape, shape)
    h, w = shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2          # matches where fftshift puts the DC term
    dist = (x - cx)**2 + (y - cy)**2
    gauss = np.exp(-(dist / (2 * sigma**2))**tp_power)
    inv_tophat = 1-gauss
    return inv_tophat


def hp_filtering_grayscale(gray, sigma_num, sobel_poly, k=3, debug=False):
    gray_part = np.multiply(sobel_poly,gray) 

    sigma = 1
    gray_blur = cv2.GaussianBlur(gray, (0, 0), sigma) #INVESTIGATE
    gray_part_blur = cv2.GaussianBlur(gray_part, (0, 0), sigma) #INVESTIGATE

    fourier_part = np.fft.fft2(gray)
    fourier_coeff = np.fft.fftshift(fourier_part)

    spectrum1 = np.log(1 + np.abs(fourier_coeff))
    spectrum1 = cv2.normalize(spectrum1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if debug:  # FIX: debug windows off by default (safe inside GUI callbacks)
        cv2.imshow("Spectrum", spectrum1)

    tp = inv_tophat(gray.shape, sigma=sigma_num)#sigma = 50, 80)
    hp_freqs = np.multiply(tp,fourier_coeff)

    spectrum2 = np.log(1 + np.abs(hp_freqs))
    spectrum2 = cv2.normalize(spectrum2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if debug:
        cv2.imshow("Spectrum2", spectrum2)

    hp = np.real(np.fft.ifft2(np.fft.ifftshift(hp_freqs)))
    #hp_norm = cv2.normalize(hp,None, 0,255,cv2.NORM_MINMAX)#.astype(np.uint8)
    #print(hp_norm.min(), hp_norm.max(),hp_norm)
    #cv2.imshow("hp2",hp_norm.astype(np.uint8))

    #Get face part of image 
    # FIX: was sobel_poly//255, which is all zeros for a 0-1 alpha
    hp_part = np.multiply(_mask01(sobel_poly),hp)

    # FIX: base on the full image, not the masked one (which blackened the background)
    gray_float = gray.astype(np.float64)
    #k = 3 #INVESTIGATE  (now a parameter)
    enhanced = gray_float+k*hp_part
    enhanced = np.clip(enhanced, 0, 255)
    if debug:
        cv2.imshow("hp_enhanced", enhanced.astype(np.uint8))
    return enhanced


def hp_filtering(channel, sigma_num, sobel_poly, k=3, debug=False):
    channel_part = np.multiply(sobel_poly,channel) 

    sigma = 1
    channel_blur = cv2.GaussianBlur(channel, (0, 0), sigma) #INVESTIGATE
    channel_part_blur = cv2.GaussianBlur(channel_part, (0, 0), sigma) #INVESTIGATE

    fourier_part = np.fft.fft2(channel)
    fourier_coeff = np.fft.fftshift(fourier_part)

    spectrum1 = np.log(1 + np.abs(fourier_coeff))
    spectrum1 = cv2.normalize(spectrum1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if debug:  # FIX: debug windows off by default (safe inside GUI callbacks)
        cv2.imshow("Spectrum", spectrum1)

    # FIX: full (h, w) shape instead of shape[0], so non-square images work
    tp = inv_tophat(channel.shape, sigma=sigma_num)#sigma = 50, 80)
    hp_freqs = np.multiply(tp,fourier_coeff)

    spectrum2 = np.log(1 + np.abs(hp_freqs))
    spectrum2 = cv2.normalize(spectrum2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if debug:
        cv2.imshow("Spectrum2", spectrum2)

    hp = np.real(np.fft.ifft2(np.fft.ifftshift(hp_freqs)))
    #hp_norm = cv2.normalize(hp,None, 0,255,cv2.NORM_MINMAX)#.astype(np.uint8)
    #print(hp_norm.min(), hp_norm.max(),hp_norm)
    #cv2.imshow("hp2",hp_norm.astype(np.uint8))

    #Get face part of image 
    hp_part = np.multiply(_mask01(sobel_poly),hp)

    channel_float = channel.astype(np.float64)
    #k = 3 #INVESTIGATE  (now a parameter, default 3)
    enhanced = channel_float+k*hp_part
    enhanced = np.clip(enhanced, 0, 255)
    if debug:
        cv2.imshow("hp_enhanced", enhanced.astype(np.uint8))
    return enhanced

def hp_filtering_bgr(img, sigma_num, sobel_poly, k=3, debug=False):

    result = np.zeros_like(img, dtype=np.float32)

    for c in range(3):

        result[:, :, c] = hp_filtering(
            img[:, :, c],
            sigma_num,
            sobel_poly,
            k=k,
            debug=debug
        )

    return result #np.clip(result, 0, 255).astype(np.uint8)

def lap_kernel_sharpen(img_bgr, sobel_poly, k=3):
    sigma = 1
    # FIX: blur only the copy the Laplacian is taken from; the base stays sharp
    img_blur = cv2.GaussianBlur(img_bgr, (0, 0), sigma) #INVESTIGATE
    lap_kernel = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])
    
    #sharp_gray = cv2.filter2D(img_bgr, ddepth = -1, kernel = sharp_kernel)
    #cv2.imshow("Gray_part2d", sharp_gray)

    sharp_layers = cv2.filter2D(img_blur,cv2.CV_32F,lap_kernel)

    #k=3  (now a parameter)
    sobel_poly3D = _mask01(sobel_poly)[...,None]
    sharp_part = np.multiply(sobel_poly3D,sharp_layers) 
    enhanced = img_bgr.astype(np.float32) + k*sharp_part

    return np.clip(enhanced, 0, 255)   # FIX: was unclipped

def sharpen(img_part, debug=False):
    sigma = 1
    #img_part = cv2.GaussianBlur(img_part, (0, 0), sigma) #INVESTIGATE
    sharp_kernel = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
    sharp_gray = cv2.filter2D(img_part, ddepth = -1, kernel = sharp_kernel)
    if debug:  # FIX: debug window off by default
        cv2.imshow("Gray_part2d", sharp_gray)
    return sharp_gray


# ------------------------------------------------------------------- blur
def apply_bokeh_effect(original_image, mask, blur_kernel_size: int = 31):

    blur_kernel_size = max(1, int(blur_kernel_size))   # FIX: guard 0 / floats
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    blurred_background = cv2.GaussianBlur(original_image, (blur_kernel_size, blur_kernel_size), 0)

    alpha = mask.astype(np.float32)
    if alpha.max() > 1:      # accept 0-255 masks too
        alpha = alpha / 255.0

    alpha_3d = np.dstack([alpha, alpha, alpha])

    # Result = Foreground * alpha + Background * (1 - alpha)
    original_float = original_image.astype(np.float32)
    background_float = blurred_background.astype(np.float32)

    portrait_float = (original_float * alpha_3d) + (background_float * (1.0 - alpha_3d))

    portrait_final = np.clip(portrait_float, 0, 255).astype(np.uint8)

    return portrait_final


def portrait_enhance(img_bgr, alpha, blur_kernel_size=31, sharpen_amount=1.0,
                     equalize_subject=False, sigma_num=50):
    """NEW (used by GUITool): sharpen (and optionally equalize) the subject,
    blur the ORIGINAL background, blend with alpha. Returns uint8 BGR.
    sharpen_amount scales the high-pass gain: 1.0 -> k=3 (previous behaviour),
    0 -> no sharpening."""
    alpha = _mask01(alpha)
    subject = img_bgr
    if equalize_subject:
        # CLAHE on lightness only, so colors are not shifted
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        subject = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    sharp = hp_filtering_bgr(subject, sigma_num, alpha, k=3 * sharpen_amount)

    blur = max(1, int(blur_kernel_size))
    if blur % 2 == 0:
        blur += 1
    background = cv2.GaussianBlur(img_bgr, (blur, blur), 0).astype(np.float32)

    a = alpha[..., None]
    result = a * sharp + (1 - a) * background
    return np.clip(result, 0, 255).astype(np.uint8)


def composite(img_bgr, preset="sobel_component", blur_kernel_size=31,
              sharpen_amount=1.0, equalize_subject=False, **overrides):
    """mask -> sharpen foreground -> blur background -> blend.
    Returns (portrait, steps). """
    p = get_params(preset, **overrides)
    alpha, steps = make_mask(img_bgr, p)
    alpha_3d = np.dstack([alpha, alpha, alpha])          # since bokeh effect on colored img
    alpha_3d = np.repeat(alpha[..., None], 3, axis=2)
    steps["sharpened"] = hp_filtering_bgr(img_bgr, sigma_num=50, sobel_poly=alpha,
                                          k=3 * sharpen_amount)#lap_kernel_sharpen(img_bgr, alpha)#sharpen(img_bgr)
    # FIX: same path as the GUI (background blurred from the original image)
    steps["portrait"] = portrait_enhance(img_bgr, alpha, blur_kernel_size,
                                         sharpen_amount, equalize_subject)
    #steps["portrait"] = apply_bokeh_effect(steps["sharpened"], alpha, 31)
    return steps["portrait"], steps


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
    # ------------------------------

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"could not read {IMAGE_PATH}")
    params = get_params(PRESET, **OVERRIDES)
    mask, steps = make_mask(img, params)

    portrait, steps = composite(img, PRESET, **OVERRIDES)   # FIX: overrides now applied
    # to_display rescales floats by their max; convert to uint8 first so the
    # sharpened preview isn't contrast-stretched
    show({"image": img, "mask": steps["alpha"],
              "sharpened": np.clip(steps["sharpened"], 0, 255).astype(np.uint8),
              "portrait": portrait}, None)
