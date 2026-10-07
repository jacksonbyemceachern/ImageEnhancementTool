"""Portrait experiment using maskutil; run directly for parameter adjustment."""
from pathlib import Path

import cv2
import numpy as np

import maskutil
from GUITool import run_GUI, make_preview


ROOT = Path(__file__).resolve().parent
# Edit these settings, then run this file directly in your IDE.
IMAGE_PATH = ROOT / "data/lfw-deepfunneled/lfw-deepfunneled/Bill_Gates/Bill_Gates_0001.jpg"
PRESET = "sobel_component"
OVERRIDES = {}  # Same configuration style as the demo in maskutil.py.
BLUR_KERNEL_SIZE = 31
SHARPEN_AMOUNT = 1.0
EQUALIZE_SUBJECT = False
OPEN_GUI = True


def run_experiment(image, params, blur=31, sharpen=1.0, equalize=False):
    """Return visual comparisons and each stage of the mask pipeline."""
    # make_mask calls gray, edges, binarize, closing, filling and mask selection.
    alpha, steps = maskutil.make_mask(image, params)
    complement = 1.0 - alpha
    blur_only = maskutil.apply_bokeh_effect(image, alpha, blur)
    # Mask after sharpening to avoid sharpening an artificial black boundary.
    sharp_image = maskutil.sharpen(image, amount=sharpen)
    a = alpha[..., None]
    sharpen_only = np.clip(np.rint(sharp_image * a + image * (1 - a)), 0, 255).astype(np.uint8)
    portrait = maskutil.portrait_enhance(image, alpha, blur, sharpen, equalize)
    equalized_portrait = maskutil.portrait_enhance(image, alpha, blur, sharpen, True)
    images = {
        "original": image,
        "subject_alpha": alpha,
        "background_alpha": complement,
        "subject_overlay": maskutil.overlay(image, alpha),
        "blur_only": blur_only,
        "sharpen_only": sharpen_only,
        "portrait": portrait,
        "portrait_CLAHE": equalized_portrait,
    }
    images["comparison"] = make_preview({
        "Original": image, "Edges": steps["edges"], "Subject alpha": alpha,
        "Background blur only": blur_only, "Subject sharpen only": sharpen_only,
        "Portrait": portrait,
    })
    images["preset_comparison"] = maskutil.compare_presets(image)
    return images, steps


def main():
    params = maskutil.get_params(PRESET, **OVERRIDES)
    if (not np.isfinite([BLUR_KERNEL_SIZE, SHARPEN_AMOUNT, params["feather_sigma"],
                         params["close_radius"], params["radial_frac"],
                         params["hull_threshold"]]).all()
            or min(BLUR_KERNEL_SIZE, SHARPEN_AMOUNT, params["feather_sigma"],
                   params["close_radius"]) < 0
            or params["radial_frac"] <= 0 or not 0 <= params["hull_threshold"] <= 255):
        raise ValueError("Use nonnegative effects, positive radial fraction and hull threshold 0-255")
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")
    images, steps = run_experiment(image, params, BLUR_KERNEL_SIZE, SHARPEN_AMOUNT, EQUALIZE_SUBJECT)
    print(f"Subject coverage: {steps['alpha'].mean():.1%}")
    if OPEN_GUI:
        return run_GUI(image, params, BLUR_KERNEL_SIZE, SHARPEN_AMOUNT,
                       EQUALIZE_SUBJECT)
    else:
        maskutil.show({"Experiment": images["comparison"]})
        return images["portrait"], steps["alpha"], steps


if __name__ == "__main__":
    main()




