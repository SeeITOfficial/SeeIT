import cv2 as cv
import numpy as np
from my_utility import select_file, reset_trackbars, save_image, help_menu

def nothing(x): pass

img = select_file()
img_copy = img.copy()
gray_img = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)

cv.namedWindow("Canny Edge Detection Controls", cv.WINDOW_NORMAL)
cv.resizeWindow("Canny Edge Detection Controls", img.shape[1], img.shape[0])

# manual canny trackbars
cv.createTrackbar("Smooth_aggrsn", "Canny Edge Detection Controls", 10, 100, nothing)
cv.createTrackbar("Smooth_region", "Canny Edge Detection Controls", 10, 100, nothing)
cv.createTrackbar("knn_pixels", "Canny Edge Detection Controls", 1, 10, nothing)
cv.createTrackbar("Kernel_size", "Canny Edge Detection Controls", 3, 7, nothing)
cv.createTrackbar("Low_thresh", "Canny Edge Detection Controls", 0, 255, nothing)
cv.createTrackbar("High_thresh", "Canny Edge Detection Controls", 127, 255, nothing)

canny_defaults = {"Low_thresh": 0, "High_thresh": 127, "Kernel_size": 3}
bilateral_defaults = {"knn_pixels": 1, "Smooth_aggrsn": 10, "Smooth_region": 10}

use_auto = False
show_help = False

while True:
    kernel_size = cv.getTrackbarPos("Kernel_size", "Canny Edge Detection Controls")
    knn_pixels = cv.getTrackbarPos("knn_pixels", "Canny Edge Detection Controls")
    smooth_aggression = cv.getTrackbarPos("Smooth_aggrsn", "Canny Edge Detection Controls")
    smooth_space = cv.getTrackbarPos("Smooth_region", "Canny Edge Detection Controls")

    if kernel_size < 3:
        kernel_size = 3
    elif kernel_size % 2 == 0:
        kernel_size += 1

    blurred_img = cv.bilateralFilter(gray_img, d=knn_pixels,
                                     sigmaColor=smooth_aggression,
                                     sigmaSpace=smooth_space)

    if use_auto:
        v = np.median(blurred_img)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        canny = cv.Canny(blurred_img, lower, upper, apertureSize=kernel_size)
        status = f"low={lower}, high={upper}"
    else:
        low_t = cv.getTrackbarPos("Low_thresh", "Canny Edge Detection Controls")
        high_t = cv.getTrackbarPos("High_thresh", "Canny Edge Detection Controls")
        canny = cv.Canny(blurred_img, low_t, high_t, apertureSize=kernel_size)
        status = ""

    stacked = np.hstack((blurred_img, canny))


    help_menu(stacked, show_help)

    cv.imshow("Canny Edge Detection", stacked)

    key = cv.waitKey(30) & 0xFF
    if key == ord('r'):
        reset_trackbars("Canny Edge Detection Controls", canny_defaults)
        reset_trackbars("Canny Edge Detection Controls", bilateral_defaults)
    elif key == ord('s'):
        save_image(stacked)
    elif key == ord('a'):
        use_auto = not use_auto
    elif key == ord('h'):
        show_help = not show_help
    elif key == 27:
        break

cv.destroyAllWindows()
