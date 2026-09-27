import sys
import cv2
import numpy as np


def head_kernel(width, height, thickness=3):
    pad = thickness
    k = np.zeros((height + 2 * pad, width + 2 * pad), np.float32)
    center = (k.shape[1] // 2, k.shape[0] // 2)
    cv2.ellipse(k, center, (width // 2, height // 2), 0, 0, 360, 1.0, thickness)
    return k / k.sum()          # normalize so kernels of any size score fairly


def find_head(edges, widths, aspect=1.3):
    best = (-1, None, None)     # (score, center, axes)
    for w in widths:
        h = int(w * aspect)
        k = head_kernel(w, h)
        if k.shape[0] > edges.shape[0] or k.shape[1] > edges.shape[1]:
            continue
        scores = cv2.matchTemplate(edges, k, cv2.TM_CCORR)
        _, score, _, (x, y) = cv2.minMaxLoc(scores)      # top-left of best match
        if score > best[0]:
            center = (x + k.shape[1] // 2, y + k.shape[0] // 2)
            best = (score, center, (w // 2, h // 2))
    return best


if __name__ == "__main__":
    img = cv2.imread('./data/lfw-deepfunneled/lfw-deepfunneled/Bill_Gates/Bill_Gates_0001.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 2), 30, 90)
    # Blur the edges so an outline that is a few pixels off still scores.
    edges = cv2.GaussianBlur(edges.astype(np.float32) / 255, (0, 0), 2)

    W = img.shape[1]
    widths = range(int(0.25 * W), int(0.65 * W), 4)    # head sizes to try

    score, center, axes = find_head(edges, widths)
    print(f"best match: center={center}, axes={axes}, score={score:.3f}")

    mask = np.zeros(gray.shape, np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)  # -1 = filled
    gradual = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 30)

    shown = img.copy()
    cv2.ellipse(shown, center, axes, 0, 0, 360, (0, 0, 255), 2)
    cv2.imshow("edges", edges / edges.max())
    cv2.imshow("best match", shown)
    cv2.imshow("mask", gradual)
    cv2.waitKey(0)
    cv2.destroyAllWindows()