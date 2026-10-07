import cv2
import numpy as np

import maskutil


def make_preview(images, columns=3):
    """Build a labeled image grid for on-screen comparison."""
    tiles = []
    h, w = next(iter(images.values())).shape[:2]
    scale = min(360 / w, 360 / h, 1.0)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    for name, image in images.items():
        tile = maskutil.to_display(image)
        if tile.ndim == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        tile = cv2.resize(tile, size, interpolation=cv2.INTER_AREA)
        tile = cv2.copyMakeBorder(tile, 30, 0, 0, 0, cv2.BORDER_CONSTANT,
                                  value=(35, 35, 35))
        cv2.putText(tile, name, (6, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + columns])
                      for i in range(0, len(tiles), columns)])


def run_GUI(image, mask_params=None, blur_kernel_size=31,
            sharpen_amount=1.0, equalize_subject=False):
    """Adjust portrait parameters and return (result, alpha, steps), without saving."""
    params = maskutil.get_params(**(mask_params or {}))
    window_name = "Portrait Adjuster"
    controls = {
        "BG blur": (round(blur_kernel_size), 101),
        "Sharpen": (round(sharpen_amount * 10), 30),
        "Feather*10": (round(params["feather_sigma"] * 10), 200),
        "Threshold": (params["threshold"] or 0, 255),
    }

    ready = False
    current = None

    def on_trackbar(_):
        nonlocal current
        # OpenCV may invoke callbacks before all sliders have been created.
        if not ready:
            return
        values = {name: cv2.getTrackbarPos(name, window_name) for name in controls}
        p = dict(params)
        p.update(feather_sigma=values["Feather*10"] / 10,
                 threshold=values["Threshold"] or None)
        alpha, steps = maskutil.make_mask(image, p)
        blur = values["BG blur"]
        result = maskutil.portrait_enhance(
            image, alpha, blur, values["Sharpen"] / 10,
            equalize_subject)
        current = result, alpha, steps
        preview = make_preview({
            "Original": image, "Edges": steps["edges"], "Subject alpha": alpha,
            "Subject overlay": maskutil.overlay(image, alpha),
            "BG blur only": maskutil.apply_bokeh_effect(image, alpha, blur),
            "Portrait": result,
        })
        cv2.imshow(window_name, preview)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(window_name, 1100, 850)
        for name, (value, maximum) in controls.items():
            cv2.createTrackbar(name, window_name, max(0, min(value, maximum)),
                               maximum, on_trackbar)
        ready = True
        on_trackbar(0)
        print("Adjust sliders. Q / Esc: close without saving.")
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        cv2.destroyAllWindows()
    return current
