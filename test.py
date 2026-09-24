import numpy as np
import cv2


img = cv2.imread('./data/lfw-deepfunneled/lfw-deepfunneled/Bill_Gates/Bill_Gates_0001.jpg', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (21,21), 3)
hpf = img - cv2.GaussianBlur(img, (21, 21), 3)+127  #127 assumed intensity of image
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

h, w = img.shape

# 1. Create a coordinate grid representing pixel locations
X, Y = np.meshgrid(np.arange(w), np.arange(h))

# 2. Define the center of the radial filter
center_x, center_y = w / 2, h / 2

# 3. Calculate distance from center for every pixel
distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

# 4. Normalize distances relative to the maximum possible distance (the corner)
max_distance = np.sqrt(center_x**2 + center_y**2)
normalized_distance = distance / (.5*max_distance)

# 5. Invert it so center is 1.0 (bright) and edge is 0.0 (dark)
# You can change the formula here to modify the intensity fall-off profile
radial_mask = 1 - normalized_distance

# Ensure values don't fall outside the 0 to 1 range
radial_mask = np.clip(radial_mask, 0, 1)

# 6. Broadcast mask dimensions to match the image channels (H, W, 1)
#radial_mask = np.expand_dims(radial_mask, axis=1)




# Compute gradient magnitude
gradient_magnitude = cv2.magnitude(sobelx, sobely)

# Convert to uint8
gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

# 7. Apply the intensity filter (convert to float math, then back to uint8)
filtered_img = (gradient_magnitude* radial_mask).astype(np.uint8)

ret, thresh_img = cv2.threshold(filtered_img, 10,70 , cv2.THRESH_BINARY)



# Display result
cv2.imshow("Sobel Edge Detection", gradient_magnitude)
cv2.imshow("image",img)
cv2.imshow("hpf",hpf)
cv2.imshow("blur",blur)
cv2.imshow("MASK", radial_mask)
cv2.imshow("filtered", filtered_img)
cv2.imshow("thresh",thresh_img)

cv2.waitKey(0)
cv2.destroyAllWindows()




