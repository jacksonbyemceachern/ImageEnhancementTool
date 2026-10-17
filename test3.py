import cv2
from GUITool import run_GUI

img = cv2.imread("path/to/image.jpg")
out = run_GUI(img, mask_params={"preset": "sobel_component"})
if out is not None:
    result, alpha, steps = out
    cv2.imwrite("portrait.png", result)