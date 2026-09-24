import numpy as np
import cv2


img = cv2.imread('./data/lfw-deepfunneled/lfw-deepfunneled/Bill_Gates/Bill_Gates_0001.jpg', cv2.IMREAD_GRAYSCALE)

f_transform = np.fft.fft2(img)
f_transform_shifted = np.fft.fftshift(f_transform)

print(f_transform)

cv2.imshow("Grayscale", img)
#cv2.imshow("Fourier", f_transform_shifted)


# Create a bandpass filter
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
inner_radius = 0.01 * crow
outer_radius = 0.1 * crow

# Create a meshgrid for the frequency coordinates
x = np.arange(-ccol, ccol)
y = np.arange(-crow, crow)
x, y = np.meshgrid(x, y)

# Create the bandpass filter
mask = ((x**2 + y**2 >= inner_radius**2) & (x**2 + y**2 <= outer_radius**2))

# Apply the mask to the shifted Fourier transform
f_transform_shifted_filtered = f_transform_shifted * mask

# Inverse Fourier Transform to get the image back
f_inverse_shifted = np.fft.ifftshift(f_transform_shifted_filtered)
image_filtered = np.fft.ifft2(f_inverse_shifted)
image_filtered = np.abs(image_filtered)

# Convert back to uint8 and normalize the values
image_filtered = np.uint8(image_filtered)
image_filtered = cv2.normalize(image_filtered, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow("Filtered", image_filtered)


cv2.waitKey(0)
cv2.destroyAllWindows()




