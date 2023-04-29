import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
path = "image.jpg"
img = cv2.imread(path, 1)

cv2.imshow('img',img)
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to separate foreground and background
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

# Perform morphological operations to remove small noise regions
kernel = np.ones((3,3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours of the foreground objects
contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Create a binary mask for the foreground
mask = np.zeros_like(closing)
cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

# Resize the mask image to match the dimensions of the original image
mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

# Apply the mask to the input image
masked_img = cv2.bitwise_and(img, img, mask=mask_resized)
masked_img1 = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
plt.imshow(masked_img1)
plt.show()

# Display the results
# cv2.imshow('Input', img)
# cv2.imshow('Thresholded', thresh)
# cv2.imshow('Closing', closing)
# cv2.imshow('Mask', mask)
# cv2.imshow('Masked Image', masked_img)

cv2.imwrite('Mask.png', mask)
cv2.imwrite('Thresholded.png', thresh)
cv2.imwrite('Closing.png', closing)
cv2.imwrite('MaskedImage.png', masked_img)

cv2.waitKey(0)