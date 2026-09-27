import cv2
import numpy as np

from PortraitMaskTool import PortraitMaskTool


def run_GUI(image, mask_tool):
    window_name = "Mask Adjuster (Press 'q' or 'ESC' to Save & Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    def on_trackbar(val):
        w_val = cv2.getTrackbarPos('Width', window_name) / 10.0
        h_val = cv2.getTrackbarPos('Height', window_name) / 10.0
        y_val = cv2.getTrackbarPos('Y-Shift', window_name) / 10.0
        b_val = cv2.getTrackbarPos('Blur', window_name) / 10.0

        preview_with_boxes, current_mask = mask_tool.generate_mask(w_scale=w_val, h_scale=h_val, y_shift_scale=y_val,
                                                                   blur_scale=b_val)

        mask_colored = np.zeros_like(image)
        mask_colored[:, :, 2] = current_mask

        overlay_preview = cv2.addWeighted(image, 0.6, mask_colored, 0.4, 0)

        raw_mask_bgr = cv2.cvtColor(current_mask, cv2.COLOR_GRAY2BGR)
        combined_view = np.hstack((overlay_preview, raw_mask_bgr))

        cv2.imshow(window_name, combined_view)

    cv2.createTrackbar('Width', window_name, 10, 30, on_trackbar)
    cv2.createTrackbar('Height', window_name, 10, 30, on_trackbar)
    cv2.createTrackbar('Y-Shift', window_name, 10, 30, on_trackbar)
    cv2.createTrackbar('Blur', window_name, 10, 50, on_trackbar)

    on_trackbar(0)

    print("Interactive window opened. Adjust sliders, then press 'q' to save and continue.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # 27 is ESC
            break

    final_w = cv2.getTrackbarPos('Width', window_name) / 10.0
    final_h = cv2.getTrackbarPos('Height', window_name) / 10.0
    final_y = cv2.getTrackbarPos('Y-Shift', window_name) / 10.0
    final_b = cv2.getTrackbarPos('Blur', window_name) / 10.0

    cv2.destroyAllWindows()

    return mask_tool.generate_mask(final_w, final_h, final_y, final_b)