import cv2 as cv
import numpy as np
from my_utility import select_file

   
def nothing(x):
    pass

image = select_file()
gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

cv.namedWindow("Controls")
cv.createTrackbar("k_size", "Controls", 3, 10, nothing)
cv.createTrackbar("Show_contours", "Controls", 0, 1, nothing)
cv.createTrackbar("Low_Thresh", 'Controls', 0, 255, nothing)
cv.createTrackbar("Struct_element", "Controls", 0, 2, nothing)

def odd_kernel(k):
    if k < 3: k = 3
    if k % 2 == 0: k += 1
    return k

structuring_element = [cv.MORPH_CROSS, cv.MORPH_ELLIPSE, cv.MORPH_RECT]


while True:
    k_size = odd_kernel(cv.getTrackbarPos("k_size", "Controls"))
    show_contours = cv.getTrackbarPos("Show_contours", "Controls")
    thresh = cv.getTrackbarPos("Low_Thresh", "Controls")
    s_elements = cv.getTrackbarPos("Struct_element", "Controls")
    
    # Binary threshold
    _, binary = cv.threshold(gray_img, thresh, 255, cv.THRESH_BINARY)
    
    kernel = cv.getStructuringElement(structuring_element[s_elements], (k_size, k_size))
    
    # Clean with morphology
    cleaned = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)  # Remove noise
    cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel)  # Fill gaps
    
    if show_contours:
        # Show why morphology matters for contour detection
        contours_raw, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours_clean, _ = cv.findContours(cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        raw_display = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
        clean_display = cv.cvtColor(cleaned, cv.COLOR_GRAY2BGR)
        
        cv.drawContours(raw_display, contours_raw, -1, (0, 255, 0), 2)
        cv.drawContours(clean_display, contours_clean, -1, (0, 255, 0), 2)
        
        # Show contour count
        cv.putText(raw_display, f"Contours: {len(contours_raw)}", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(clean_display, f"Contours: {len(contours_clean)}", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        stacked = np.hstack((raw_display, clean_display))
        cv.imshow("Raw vs Cleaned Contours", stacked)
    else:
        stacked = np.hstack((binary, cleaned))
        cv.imshow("Before vs After Morphology", stacked)
    
    key = cv.waitKey(30) & 0xFF
    if key == 27: break

cv.destroyAllWindows()