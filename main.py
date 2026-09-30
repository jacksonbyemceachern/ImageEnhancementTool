import maskutil as pm
import cv2

img = cv2.imread("./data/lfw-deepfunneled/lfw-deepfunneled/Bill_Gates/Bill_Gates_0001.jpg")


mask, complement = pm.mask_and_complement(img, "sobel_hull", close_radius=5, feather_sigma=3)

cv2.imshow("mask", mask)
cv2.imshow("complement", complement)
cv2.waitKey(0)
cv2.destroyAllWindows()