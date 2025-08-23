import cv2 as cv
import numpy as np
from my_utility import select_file, save_image, cvt2gray, reset_trackbars

def nothing(x):
    pass

# Load image
img = select_file()

# Convert to grayscale
gray = cvt2gray(img)

# Ensure image is at least 15 cm wide and tall 
dpi = 96
min_size = int((10 / 2.54) * dpi)
h, w = img.shape[:2]
scale_factor = max(min_size / w, min_size / h, 1.0)
if scale_factor > 1.0:
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_CUBIC)
    gray = cvt2gray(img)  # resize gray too
    cv.putText(img, "Resized", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)

# Default parameters
harris_defaults = {
    'block': 2,
    'ksize': 3,  # must be odd 
    'sensitivity': 0.04     
}
shi_defaults = {
    'max_corners': 100,
    'quality': 0.01,
    'min_dist': 10
}

# Control window
cv.namedWindow("Controls", cv.WINDOW_NORMAL)
cv.resizeWindow("Controls", max(img.shape[1], min_size), max(img.shape[0], min_size))

# Trackbars
cv.createTrackbar("Block_size", "Controls", harris_defaults['block'], 20, nothing)
cv.createTrackbar("Ksize", "Controls", harris_defaults['ksize'], 20, nothing)
cv.createTrackbar("Sensitivity x1000", "Controls", int(harris_defaults['sensitivity'] * 1000), 100, nothing)

cv.createTrackbar("Shi MaxCorners", "Controls", shi_defaults['max_corners'], 1000, nothing)
cv.createTrackbar("Shi Quality x100", "Controls", int(shi_defaults['quality'] * 100), 100, nothing)
cv.createTrackbar("Shi MinDist", "Controls", shi_defaults['min_dist'], 100, nothing)

while True:
    # Harris params
    block_size = max(2, cv.getTrackbarPos("Block_size", "Controls"))
    kernel_size = cv.getTrackbarPos("Ksize", "Controls")
    kernel_size = kernel_size if kernel_size % 2 == 1 and kernel_size >= 3 else 3
    raw_val = cv.getTrackbarPos("Sensitivity x1000", "Controls")
    sensitivity = raw_val / 1000.0
    sensitivity = np.clip(sensitivity, 0.01, 0.1)

    # Shi-Tomasi params
    max_corners = max(1, cv.getTrackbarPos("Shi MaxCorners", "Controls"))
    corner_quality = max(0.01, cv.getTrackbarPos("Shi Quality x100", "Controls") / 100.0)
    min_distance = max(1, cv.getTrackbarPos("Shi MinDist", "Controls"))

    # Harris detection
    harris_response = cv.cornerHarris(gray, block_size, kernel_size, sensitivity)
    # Dilate to find local maxima
    dilated = cv.dilate(harris_response, None)
    local_max = (harris_response == dilated)  # keep only peaks

    # Apply threshold
    thresh = 0.01 * harris_response.max()
    harris_points = np.argwhere((harris_response > thresh) & local_max)

    harris_img = img.copy()
    for y, x in harris_points:
        cv.circle(harris_img, (x, y), 2, (0, 0, 255), -1)  # radius=1, solid dot

    # Shi-Tomasi detection
    shi_img = img.copy()
    shi_corners = cv.goodFeaturesToTrack(gray, max_corners, corner_quality, min_distance)
    if shi_corners is not None:
        for c in shi_corners:
            x, y = c.ravel()
            cv.circle(shi_img, (int(x), int(y)), 2, (0, 255, 0), -1)

    # Display side-by-side
    combined = np.hstack((harris_img, shi_img))
    cv.imshow("Harris vs Shi-Tomasi", combined)

    key = cv.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('s'):  # Save
        save_image("Harris_vs_Shi.png", combined)
    elif key == ord('r'):  # Reset
        reset_trackbars("Controls", (
            harris_defaults['block'],
            harris_defaults['ksize'],
            int(harris_defaults['sensitivity'] * 100),
            shi_defaults['max_corners'],
            int(shi_defaults['quality'] * 100),
            shi_defaults['min_dist']
        ))

cv.destroyAllWindows()
