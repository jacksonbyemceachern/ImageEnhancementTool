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
    params = maskutil.get_params(**(mask_params or {}))
    window_name = "Portrait Adjuster"
    controls = {
        "BG blur":     (round(blur_kernel_size), 101),
        "Sharpen":     (round(sharpen_amount * 10), 30),
        "Feather*10":  (round(params["feather_sigma"] * 10), 200),
        "Threshold":   (params["threshold"] or 0, 255),          # 0 = Otsu
        "Canny":       (int(params["edges"] == "canny"), 1),
        "Close r":     (params["close_radius"], 10),
        "Hull":        (int(params["method"] == "hull"), 1),     # 0 = component
        "Radial %":    (round(params["radial_frac"] * 100), 100),
        "Laplacian":   (0, 1),                                   # 0 = Gaussian HP
        "k*10":        (50, 100),
        "HP D0":       (50, 125),
        "Pre-blur*10": (10, 30),
        "Equalize":    (int(equalize_subject), 1),
    }

    ready = False
    current = None
    shown = {}  # the images currently displayed in the preview grid

    def on_trackbar(_):
        nonlocal current, shown
        # OpenCV may invoke callbacks before all sliders have been created.
        if not ready:
            return
        values = {name: cv2.getTrackbarPos(name, controls_window) for name in controls}
        p = dict(params)
        p.update(feather_sigma=values["Feather*10"] / 10,
                 threshold=values["Threshold"] or None,
                 edges="canny" if values["Canny"] else "sobel",
                 close_radius=values["Close r"],
                 method="hull" if values["Hull"] else "component",
                 radial_frac=max(1, values["Radial %"]) / 100,
                 )
        alpha, steps = maskutil.make_mask(image, p)
        blur = values["BG blur"]
        result = maskutil.portrait_enhance(
            image, alpha, blur, values["Sharpen"] / 10,
            equalize_subject=bool(values["Equalize"]),
            sigma_num=max(1, values["HP D0"]), laplacian=bool(values["Laplacian"]),
            k=values["k*10"] / 10, pre_blur=values["Pre-blur*10"] / 10)
        current = result, alpha, steps
        shown = {
            "Original": image, "Edges": steps["edges"], "Subject alpha": alpha,
            "Subject overlay": maskutil.overlay(image, alpha),
            "BG blur only": maskutil.apply_bokeh_effect(image, alpha, blur),
            "Portrait": result,
        }
        cv2.imshow(window_name, make_preview(shown))

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
        print("Adjust sliders. S: save all shown images. Q / Esc: close.")
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s") and shown:
                # Each tile as its own PNG, plus the whole grid.
                for name, im in shown.items():
                    cv2.imwrite(f"{name.replace(' ', '_')}.png", maskutil.to_display(im))
                cv2.imwrite("preview.png", make_preview(shown))
                print("Saved", ", ".join(shown), "and preview.png")
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
    #data\lfw-deepfunneled\lfw-deepfunneled\Scott_Peterson\Scott_Peterson_0002.jpg
    #IMAGE_PATH = "./data/lfw-deepfunneled/lfw-deepfunneled/Bill_Gates/Bill_Gates_0001.jpg"
    #IMAGE_PATH = "./data/lfw-deepfunneled/lfw-deepfunneled/Abel_Aguilar/Abel_Aguilar_0001.jpg"
    #IMAGE_PATH = "./data/lfw-deepfunneled/lfw-deepfunneled/Zarai_Toledo/Zarai_Toledo_0001.jpg" 
    #IMAGE_PATH = "./data/lfw-deepfunneled/lfw-deepfunneled/Zoran_Djindjic/Zoran_Djindjic_0001.jpg"
    #IMAGE_PATH = "./data/lfw-deepfunneled/lfw-deepfunneled/Ziwang_Xu/Ziwang_Xu_0001.jpg"
    IMAGE_PATH = "./data/lfw-deepfunneled/lfw-deepfunneled/Scott_Peterson/Scott_Peterson_0002.jpg"
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"could not read {IMAGE_PATH}")
    out = run_GUI(img, mask_params={"preset": "sobel_component"})
    if out is not None:
        result, alpha, steps = out