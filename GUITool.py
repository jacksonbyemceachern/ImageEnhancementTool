import cv2
import numpy as np

import maskutil

#Set up controls window - from AI IDK how this gui works
controls = {
    "BG blur":     (31, 101),
    "Feather*10":  (100, 200),
    "Threshold":   (50, 255),    # 0 = Otsu
    "Canny":       (0, 1),
    "Close r":     (3, 10),
    "Hull":        (0, 1),       # 0 = component
    "Radial %":    (50, 100),
    "Laplacian":   (0, 1),       # 0 = Gaussian HP
    "k*10":        (50, 100),
    "HP D0":       (50, 125),    # ignored when Laplacian is on
    "Pre-blur*10": (10, 30),
    "Equalize":    (0, 1),
}


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
    """Adjust portrait parameters and return (result, alpha, steps), without saving.
    mask_params may include "preset" plus any maskutil.DEFAULTS overrides,
    e.g. {"preset": "sobel_hull", "close_radius": 5}."""
    params = maskutil.get_params(**(mask_params or {}))
    window_name = "Portrait Adjuster"
    controls = {
    "BG blur":     (round(blur_kernel_size), 101),
    "Sharpen": (round(sharpen_amount * 10), 30),
    "Feather*10":  (round(params["feather_sigma"] * 10), 200),
    "Threshold":   (params["threshold"] or 0, 255),              # 0 = Otsu
    "Canny":       (int(params["edges"] == "canny"), 1),
    "Close r":     (params["close_radius"], 10),
    "Hull":        (int(params["method"] == "hull"), 1),         # 0 = component
    "Radial %":    (round(params["radial_frac"] * 100), 100),
    "Laplacian":   (0, 1),                                       # 0 = Gaussian HP
    "k*10":        (50, 100),
    "HP D0":       (50, 125),
    "Pre-blur*10": (10, 30),
    "Equalize":    (int(equalize_subject), 1),
}

    ready = False
    current = None

    def on_trackbar(_):
        nonlocal current
        # OpenCV may invoke callbacks before all sliders have been created.
        if not ready:
            return
        values = {name: cv2.getTrackbarPos(name, controls_window) for name in controls}
        p = dict(params)
        p.update(feather_sigma=values["Feather*10"] / 10,
                 threshold=values["Threshold"] or None,
                 edges="canny" if values["Canny"] else "sobel",
                 close_radius=values["Close r"],
                 method = "hull" if values["Hull"] else "component",
                 radial_frac = max(1, values["Radial %"]),
                 )
        alpha, steps = maskutil.make_mask(image, p)
        blur = values["BG blur"]
        result = maskutil.portrait_enhance(
            image, alpha, blur, values["Sharpen"] / 10,
            equalize_subject=bool(values["Equalize"]),
            sigma_num=max(1, values["HP D0"]), laplacian=bool(values["Laplacian"]),
             k=values["k*10"] / 10, pre_blur = values["Pre-blur*10"] /10 )
        current = result, alpha, steps
        preview = make_preview({
            "Original": image, "Edges": steps["edges"], "Subject alpha": alpha,
            "Subject overlay": maskutil.overlay(image, alpha),
            "BG blur only": maskutil.apply_bokeh_effect(image, alpha, blur),
            "Portrait": result,
        })
        cv2.imshow(window_name, preview)

    controls_window = "Controls"

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    cv2.namedWindow(controls_window, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(controls_window, 420, 40 * len(controls))
        cv2.imshow(controls_window, np.zeros((1, 420, 3), np.uint8))  # some backends need an image to draw the window
        cv2.moveWindow(controls_window, 0, 0)
        cv2.moveWindow(window_name, 440, 0)                            # preview to the right of the controls

        for name, (value, maximum) in controls.items():
            cv2.createTrackbar(name, controls_window, max(0, min(value, maximum)),
                                maximum, on_trackbar)
        ready = True
        on_trackbar(0)

        print("Sharpen / Feather values are divided by 10; Threshold 0 selects Otsu.")
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


if __name__ == "__main__":
    # Standalone entry point, so the GUI can be run without a separate caller.
    #data\lfw-deepfunneled\lfw-deepfunneled\Zoran_Djindjic\Zoran_Djindjic_0001.jpg
    IMAGE_PATH = "./data/lfw-deepfunneled/lfw-deepfunneled/Bill_Gates/Bill_Gates_0001.jpg"
    IMAGE_PATH = "./data/lfw-deepfunneled/lfw-deepfunneled/Abel_Aguilar/Abel_Aguilar_0001.jpg"
    IMAGE_PATH = "./data/lfw-deepfunneled/lfw-deepfunneled/Zarai_Toledo/Zarai_Toledo_0001.jpg" 
    IMAGE_PATH = "./data/lfw-deepfunneled/lfw-deepfunneled/Zoran_Djindjic\Zoran_Djindjic_0001.jpg" 
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"could not read {IMAGE_PATH}")
    out = run_GUI(img, mask_params={"preset": "sobel_component"})
    if out is not None:
        result, alpha, steps = out
        # cv2.imwrite("portrait.png", result)   # uncomment to save
