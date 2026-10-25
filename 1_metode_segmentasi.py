import cv2
import numpy as np

def proses_thresholding(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def proses_hue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Range HSV untuk warna yang cukup pekat
    lower_mask = np.array([0, 50, 50])
    upper_mask = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_mask, upper_mask)
    return cv2.bitwise_and(img, img, mask=mask)

def proses_tepi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

def proses_hough(img):
    result = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=20, maxRadius=200)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 3)
            cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)
    return result