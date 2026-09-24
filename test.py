
import cv2


img = cv2.imread('./data/lfw-deepfunneled/lfw-deepfunneled/Bill_Gates/Bill_Gates_0001.jpg', cv2.IMREAD_GRAYSCALE)
hpf = img - cv2.GaussianBlur(img, (21, 21), 3)+127
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

# Compute gradient magnitude
gradient_magnitude = cv2.magnitude(sobelx, sobely)

# Convert to uint8
gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

# Display result
cv2.imshow("Sobel Edge Detection", gradient_magnitude)
cv2.imshow("image",img)
cv2.imshow("hpf",hpf)

cv2.waitKey(0)
cv2.destroyAllWindows()


