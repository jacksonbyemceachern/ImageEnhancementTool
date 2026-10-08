import cv2
import numpy as np


def make_mask(img, threshold=50, radial_frac=0.5, hull_threshold=50):
    """Returns a mask that is 1 on the subject and 0 on the background."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel edge strength
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    edges = cv2.convertScaleAbs(cv2.magnitude(gx, gy))

    # Threshold, then fill regions enclosed by edges
    _, binary = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)
    flood = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(flood, None, (0, 0), 255)
    filled = binary | ~flood[1:-1, 1:-1]

    # Weight: 1 at the center, 0 at radial_frac of the way to the corner
    h, w = gray.shape
    Y, X = np.mgrid[:h, :w]
    dist = np.hypot(X - w / 2, Y - h / 2)
    radial = np.clip(1 - dist / (radial_frac * np.hypot(w / 2, h / 2)), 0, 1)

    # Convex hull around the strong edges near the center
    weighted = edges * radial * (filled > 0)
    points = cv2.findNonZero((weighted > hull_threshold).astype(np.uint8))
    mask = np.zeros((h, w), np.float32)
    if points is not None:
        cv2.fillPoly(mask, [cv2.convexHull(points)], 1.0)
    return mask


def sharpen(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def portrait(img, blur_size=31):
    """Sharp subject over a blurred background. blur_size must be odd."""
    mask = make_mask(img)[..., None]          # (h, w) -> (h, w, 1)
    sharp = sharpen(img)
    blurred = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    result = mask * sharp + (1 - mask) * blurred
    return result.astype(np.uint8), mask


img = cv2.imread("./data/lfw-deepfunneled/lfw-deepfunneled/Bill_Gates/Bill_Gates_0001.jpg")
result, mask = portrait(img)

cv2.imshow("image", img)
cv2.imshow("mask", mask)
cv2.imshow("portrait", result)
cv2.waitKey(0)
cv2.destroyAllWindows()