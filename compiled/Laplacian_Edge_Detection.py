import cv2 as cv
import numpy as np
from my_utility import select_file, save_image, reset_trackbars, help_menu

def nothing(x):
    pass

img = select_file()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.namedWindow("Laplacian Edge Detector Controls", cv.WINDOW_NORMAL)
cv.resizeWindow("Laplacian Edge Detector Controls", img.shape[1], img.shape[0])

cv.createTrackbar("Gauss_K_size", "Laplacian Edge Detector Controls", 3, 9, nothing)
cv.createTrackbar("Lap_K_size", "Laplacian Edge Detector Controls", 3, 9, nothing)

k_defaults = {"Gauss_K_size": 3, "Lap_K_size": 3}

show_help = False

while True:
    Gauss_kernel_size = cv.getTrackbarPos("Gauss_K_size", "Laplacian Edge Detector Controls")
    Lap_kernel_size   = cv.getTrackbarPos("Lap_K_size", "Laplacian Edge Detector Controls")

    # Enforce minimum odd sizes
    Gauss_kernel_size = max(3, Gauss_kernel_size | 1)
    Lap_kernel_size   = max(3, Lap_kernel_size   | 1)

    blurred = cv.GaussianBlur(gray, (Gauss_kernel_size, Gauss_kernel_size), 0)

    lap_float   = cv.Laplacian(blurred, cv.CV_64F, ksize=Lap_kernel_size, scale=1.0)
    lap_display = cv.normalize(lap_float, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    Display = np.hstack((blurred, lap_display))

    help_menu(Display, show_help)
    cv.imshow("Laplacian Edge Detector", Display)

    key = cv.waitKey(30) & 0xFF
    if key == ord('s'):
        save_image(Display)
    elif key == ord('h'):
        show_help = not show_help
    elif key == ord('r'):
        reset_trackbars("Laplacian Edge Detector Controls", k_defaults)  
    elif key == 27:
        break

cv.destroyAllWindows()
