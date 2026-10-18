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


def run_GUI(image, mask_params=None, blur_kernel_size=31):
    params = maskutil.get_params(**(mask_params or {}))
    window_name = "Portrait Adjuster"
    # Short slider names, because OpenCV truncates long trackbar labels.
    # The full description of each is drawn in the Controls window (KEY).
    controls = {
        "BG blur":   (round(blur_kernel_size), 101),
        "Feather":   (round(params["feather_sigma"] * 10), 200),
        "Canny":     (int(params["edges"] == "canny"), 1),  # 0 = Sobel
        "T(Sobel)":  (params["threshold"], 255),
        "Close r":   (params["close_radius"], 10),
        "Laplace":   (0, 1),                                # 0 = Gaussian HP
        "k":         (50, 100),
        "D0(HP)":    (50, 125),
    }
    KEY = [
        ("BG blur",  "background Gaussian kernel size"),
        ("Feather",  "mask feather sigma x10"),
        ("Canny",    "0 = Sobel, 1 = Canny edges"),
        ("T(Sobel)", "edge threshold - Sobel only"),
        ("Close r",  "closing disk radius"),
        ("Laplace",  "0 = Gaussian high-pass, 1 = Laplacian"),
        ("k",        "sharpening strength x10"),
        ("D0(HP)",   "high-pass cutoff D0 - HP only"),
    ]

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
        p.update(feather_sigma=values["Feather"] / 10,
                 threshold=max(1, values["T(Sobel)"]),
                 edges="canny" if values["Canny"] else "sobel",
                 close_radius=values["Close r"],
                 )
        alpha, steps = maskutil.make_mask(image, p)
        blur = values["BG blur"]
        result = maskutil.portrait_enhance(
            image, alpha, blur,
            sigma_num=max(1, values["D0(HP)"]), laplacian=bool(values["Laplace"]),
            k=values["k"] / 10)   # pre-blur fixed at sigma = 1 (report III.G)
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

    # AUTOSIZE: the window takes the size of the key image, so the key is not stretched
    cv2.namedWindow(controls_window, cv2.WINDOW_AUTOSIZE)
    try:
        key_img = np.full((22 * len(KEY) + 12, 420, 3), 35, np.uint8)
        for i, (name, text) in enumerate(KEY):
            y = 22 * i + 24
            cv2.putText(key_img, name, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(key_img, text, (95, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow(controls_window, key_img)
        cv2.moveWindow(controls_window, 0, 0)
        cv2.moveWindow(window_name, 440, 0)                            # preview to the right of the controls

        for name, (value, maximum) in controls.items():
            cv2.createTrackbar(name, controls_window, max(0, min(value, maximum)),
                               maximum, on_trackbar)
        ready = True
        on_trackbar(0)

        print("Feather and k values are divided by 10.")
        print("Canny: 0 = Sobel, 1 = Canny (T(Sobel) applies to Sobel only).")
        print("Laplace: 0 = Gaussian high-pass, 1 = Laplacian kernel (D0(HP) applies to HP only).")
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
    out = run_GUI(img)
    if out is not None:
        result, alpha, steps = out