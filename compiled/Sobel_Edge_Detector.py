import cv2 as cv
import numpy as np
from my_utility import select_file, save_image, reset_trackbars, help_menu

def nothing(x):
    pass

def ensure_odd(k):
    if k < 3: 
        k = 3
    if k % 2 == 0:
        k += 1
    return k

img = select_file()
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.namedWindow("Sobel Edge Detector Controls", cv.WINDOW_NORMAL)
cv.resizeWindow("Sobel Edge Detector Controls", img.shape[1], img.shape[0])

cv.createTrackbar("Gauss_k_size", "Sobel Edge Detector Controls", 3, 10, nothing)
cv.createTrackbar("SobelX_k_size", "Sobel Edge Detector Controls", 3, 10, nothing)
cv.createTrackbar("SobelY_k_size", "Sobel Edge Detector Controls", 3, 10, nothing)

kernel_defaults = {"Gauss_k_size": 3,
                   "SobelX_k_size": 3,
                   "SobelY_k_size": 3}

show_help = False

while True:
    # Trackbar values
    gauss_kernel = ensure_odd(cv.getTrackbarPos("Gauss_k_size", "Sobel Edge Detector Controls"))
    sobelX_kernel = ensure_odd(cv.getTrackbarPos("SobelX_k_size", "Sobel Edge Detector Controls"))
    sobelY_kernel = ensure_odd(cv.getTrackbarPos("SobelY_k_size", "Sobel Edge Detector Controls"))

    # Blur first (denoise)
    blurred = cv.GaussianBlur(gray_img, (gauss_kernel, gauss_kernel), 0)

    # Sobel gradients
    grad_x = cv.Sobel(blurred, cv.CV_32F, 1, 0, ksize=sobelX_kernel, scale=1.0/sobelX_kernel)
    grad_y = cv.Sobel(blurred, cv.CV_32F, 0, 1, ksize=sobelY_kernel, scale=1.0/sobelY_kernel)

    # Gradient magnitude
    magnitude = cv.magnitude(grad_x, grad_y)
    min_val, max_val = np.min(magnitude), np.max(magnitude)
    if max_val > 0:
        Sobel_norm = ((magnitude - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        Sobel_norm = np.zeros_like(magnitude, dtype=np.uint8)



    # Stack for display
    stacked = np.hstack((blurred, Sobel_norm))

    help_menu(stacked, show_help)
    cv.imshow("Sobel Edge Detector", Sobel_norm)

    key = cv.waitKey(30) & 0xFF
    if key == ord('s'):
        save_image(stacked)
    elif key == ord('r'):
        reset_trackbars("Sobel Edge Detector Controls", kernel_defaults)
    elif key == ord('h'):
        show_help = not show_help
    elif key == 27:  # Esc
        break

cv.destroyAllWindows()
