import cv2
import numpy as np


class PortraitMaskTool:
    def __init__(self, image, faces: list):
        self.image = image
        self.faces = faces

    def generate_mask(self, w_scale=1.0, h_scale=1.0, y_shift_scale=1.0, mask_blur_scale=1.0):

        height, width = self.image.shape[:2]
        hard_mask = np.zeros((height, width), dtype=np.uint8)

        for (x, y, w, h) in self.faces:
            center_x = x + w // 2
            center_y = min(height - 1, y + int(h * 0.75 * y_shift_scale))
            center = (center_x, center_y)

            axis_w = max(1, int(w * 0.90 * w_scale))
            axis_h = max(1, int(h * 1.35 * h_scale))
            axes = (axis_w, axis_h)

            cv2.ellipse(hard_mask, center, axes, 0, 0, 360, 255, thickness=cv2.FILLED)

        sigma = max(2, min(height, width) / 90) * mask_blur_scale
        if sigma <= 0:
            return hard_mask

        return cv2.GaussianBlur(hard_mask, (0, 0), sigmaX=sigma)