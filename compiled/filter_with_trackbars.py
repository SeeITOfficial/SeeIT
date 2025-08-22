import cv2 as cv
import numpy as np
from my_utility import select_file, save_image, reset_trackbars


def nothing(x):
    pass

# Get the image (already resized to fit screen)
img = select_file()
img_copy = img.copy()

# Create separate control window to save image space
cv.namedWindow("Color Filter", cv.WINDOW_NORMAL)
cv.namedWindow("Controls", cv.WINDOW_NORMAL)
cv.resizeWindow("Color Filter", img.shape[1], img.shape[0])
cv.resizeWindow("Controls", 400, 300)

# Trackbars
cv.createTrackbar("R", "Controls", 100, 200, nothing)
cv.createTrackbar("G", "Controls", 100, 200, nothing) 
cv.createTrackbar("B", "Controls", 100, 200, nothing)

cv.createTrackbar("L", "Controls", 100, 200, nothing)
cv.createTrackbar("A", "Controls", 100, 200, nothing)
cv.createTrackbar("B*", "Controls", 100, 200, nothing)

cv.createTrackbar("H", "Controls", 90, 180, nothing)
cv.createTrackbar("S", "Controls", 100, 200, nothing)
cv.createTrackbar("V", "Controls", 100, 200, nothing)

# Pre-convert color spaces
base_rgb = cv.cvtColor(img_copy, cv.COLOR_BGR2RGB).astype(np.float32)
base_lab = cv.cvtColor(img_copy, cv.COLOR_BGR2LAB).astype(np.float32)
base_hsv = cv.cvtColor(img_copy, cv.COLOR_BGR2HSV).astype(np.float32)

color_defaults = {
        "R": 100,
        "G": 100,
        "B": 100,
        "L": 100,
        "A": 100,
        "B*": 100,
        "H": 90,
        "S": 100,
        "V": 100
    }



show_help = False  # toggle flag

def draw_help_menu(frame):
    help_text = [
        "r : Reset sliders",
        "s : Save image",
        "ESC : Exit"
    ]
    y0, dy = 20, 25
    for i, line in enumerate(help_text):
        y = y0 + i * dy
        cv.putText(frame, line, (10, y), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2, cv.LINE_AA)

def draw_press_h(frame):
    cv.putText(frame, "Press H for Help", (10, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

while True:
    rgb_img = base_rgb.copy()
    lab_img = base_lab.copy()
    hsv_img = base_hsv.copy()

    rgb_red = cv.getTrackbarPos("R", "Controls") / 100.0
    rgb_green = cv.getTrackbarPos("G", "Controls") / 100.0
    rgb_blue = cv.getTrackbarPos("B", "Controls") / 100.0

    lab_lightness = cv.getTrackbarPos("L", "Controls") / 100.0
    lab_greenRed = cv.getTrackbarPos("A", "Controls") / 100.0
    lab_blueYellow = cv.getTrackbarPos("B*", "Controls") / 100.0

    hsv_hue = cv.getTrackbarPos("H", "Controls") - 90
    hsv_sat = cv.getTrackbarPos("S", "Controls") / 100.0
    hsv_value = cv.getTrackbarPos("V", "Controls") / 100.0

    # Apply RGB
    rgb_img[:, :, 0] *= rgb_red
    rgb_img[:, :, 1] *= rgb_green
    rgb_img[:, :, 2] *= rgb_blue

    # Apply LAB
    lab_img[:, :, 0] *= lab_lightness
    lab_img[:, :, 1] = 128 + (lab_img[:, :, 1] - 128) * lab_greenRed
    lab_img[:, :, 2] = 128 + (lab_img[:, :, 2] - 128) * lab_blueYellow

    # Apply HSV
    hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hsv_hue) % 180
    hsv_img[:, :, 1] *= hsv_sat
    hsv_img[:, :, 2] *= hsv_value

    # Back to BGR
    bgr_from_rgb = cv.cvtColor(np.clip(rgb_img, 0, 255).astype(np.uint8), cv.COLOR_RGB2BGR)
    bgr_from_hsv = cv.cvtColor(np.clip(hsv_img, 0, 255).astype(np.uint8), cv.COLOR_HSV2BGR)

    lab_clipped = lab_img.copy()
    lab_clipped[:, :, 0] = np.clip(lab_clipped[:, :, 0], 0, 100)
    lab_clipped[:, :, 1] = np.clip(lab_clipped[:, :, 1], 0, 255)
    lab_clipped[:, :, 2] = np.clip(lab_clipped[:, :, 2], 0, 255)
    bgr_from_lab = cv.cvtColor(lab_clipped.astype(np.uint8), cv.COLOR_LAB2BGR)

    final_bgr = ((bgr_from_rgb.astype(np.float32) +
                  bgr_from_hsv.astype(np.float32) +
                  bgr_from_lab.astype(np.float32)) / 3).astype(np.uint8)

    # Draw overlay on the frame being displayed
    if show_help:
        draw_help_menu(final_bgr)
    else:
        draw_press_h(final_bgr)

    cv.imshow("Color Filter", final_bgr)

    key = cv.waitKey(30) & 0xFF
    if key == ord('h'):
        show_help = not show_help
    elif key == ord('r'):
        reset_trackbars("Color Filters", color_defaults)
    elif key == ord('s'):
        save_image(final_bgr)
    elif key == 27:
        break

cv.destroyAllWindows()
