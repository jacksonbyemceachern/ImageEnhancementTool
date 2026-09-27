import cv2
import numpy as np

def _disk(radius):
    """Circular structuring element of the given radius."""
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


if __name__ == "__main__":
    # Demo: python mask.py path/to/image.jpg
    import sys
    import matplotlib.pyplot as plt

    #Import Image
    img_bgr = cv2.imread('./data/lfw-deepfunneled/lfw-deepfunneled/Bill_Gates/Bill_Gates_0001.jpg')
    print(img_bgr.shape)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)   # uint8
    #Blurring??
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    #Edge Detection
    canny_edges = cv2.Canny(gray, 50, 150)  
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges 
    sobel_edges = cv2.magnitude(sobelx,sobely)
    sobel_edges = cv2.convertScaleAbs(sobel_edges)


    #tinker with edges to work better
    sobel_scaled = cv2.normalize(sobel_edges.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sobel_binary = cv2.threshold(sobel_scaled,50,255,cv2.THRESH_BINARY)
    #What is morphologyEx???
    sobel_closed = cv2.morphologyEx(sobel_binary, cv2.MORPH_CLOSE, _disk(3))



    #Flood Fill
    padded = cv2.copyMakeBorder(sobel_binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ff_mask = np.zeros((250 + 4, 250 + 4), np.uint8)   # im height is 250
    #HOW DOES THIS WORK vvv
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    sobel_flooded = cv2.bitwise_or(sobel_binary, holes)


    #Add radial mask filter here???:
    h, w = sobel_flooded.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    center_x, center_y = w / 2, h / 2
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    normalized_distance = distance / (.5*max_distance)
    radial_mask = 1 - normalized_distance
    radial_mask = np.clip(radial_mask, 0, 1)

    radial_sobel = np.multiply(sobel_flooded, radial_mask)

    #Select Head
    ##HOW DOES THIS WORK???
    ## TINKER W/ Connectivity???
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(radial_sobel.astype(np.uint8), connectivity=8)
    h, w = sobel_flooded.shape
    cx, cy = (w - 1) / 2, (h - 1) / 2
    center_label = labels[int(round(cy)), int(round(cx))]
    mask = np.where(labels == center_label, 255, 0).astype(np.uint8)


    ### Contours Here?
    _, c_sobel_binary = cv2.threshold(np.multiply(radial_sobel, sobel_scaled).astype(np.uint8), 50,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(c_sobel_binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.vstack(contours))
    out = np.zeros_like(sobel_binary)
    sobel_poly = cv2.fillPoly(out, [hull], 255)



    
    cv2.imshow("Canny",canny_edges)
    cv2.imshow("Sobel", sobel_edges)
    cv2.imshow("Radial Mask", radial_mask)
    cv2.imshow("Radial_Sobel", radial_sobel)
    cv2.imshow("Scaled Sobel", sobel_scaled)
    cv2.imshow("Sobel Binary", sobel_binary)
    cv2.imshow("Sobel Closed", sobel_closed)
    cv2.imshow("Flood", flood)
    cv2.imshow("holes", holes)
    cv2.imshow("Sobel Flooded", sobel_flooded)
    cv2.imshow("Labels", labels.astype(np.uint8))
    cv2.imshow("Centered Sobel Binary", c_sobel_binary)
    cv2.imshow("Sobel Contouring", sobel_poly)

    # for i in range(1, n):                      # skip label 0 (background)
    #     area = stats[i, cv2.CC_STAT_AREA]
    #     component = np.where(labels == i, 255, 0).astype(np.uint8)

    #     cv2.imshow("component", component)
    #     cv2.setWindowTitle("component", f"Label {i} of {n - 1}, area {area}")

    #     key = cv2.waitKey(0)                   # wait for any key
    #     if key == ord("q"):                    # press q to stop early
    #         break

    # cv2.destroyAllWindows()

    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

