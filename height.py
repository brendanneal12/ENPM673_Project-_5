import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Estimates the water level and returns the height in pixels
def water_level(frame):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    smooth_frame = cv.GaussianBlur(gray_frame, (5,5), 0) # reduce noise
    _, thresh_frame = cv.threshold(smooth_frame, 215, 255, cv.THRESH_BINARY) # binary threshold

    foreground_frame= back_subtract.apply(thresh_frame) # separate the background and foreground

    contours, hierarchy = cv.findContours(foreground_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # find contours

    y = 0
    if len(contours) > 0:
        cont_max = max(contours, key = cv.contourArea)  # get the contour with the biggest area
        x, y, w, h = cv.boundingRect(cont_max)
        #cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv.line(frame, (x, y), (x+150, y), (0, 255, 255), 3)  # draw horizontal line at the water height level

    #cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
    cv.imshow('Water Level', frame)
    return y
    

ship_vid = cv.VideoCapture("Vessel Draft Video.mp4") # read the video

back_subtract = cv.createBackgroundSubtractorMOG2()

while ship_vid.isOpened():
    ret, frame = ship_vid.read()

    if ret == True:
        height = water_level(frame)
        if height != 0:
            print("Water level: " + str(height) + " pixels")
    
        if cv.waitKey(25) & 0xFF == ord('e'):
            break
    else:
        break

ship_vid.release()
cv.destroyAllWindows()