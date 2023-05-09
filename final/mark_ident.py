import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def levelEst(markings, lvl, pxl_avg):
    mark_dict = {}
    vals = []
    height = 8.2

    #creating marking dictionary for pixel to meter translation
    for i in range(0, len(markings)):
        vals.append(height)
        height = round((height + 0.1), 2)
    for i in range(0, len(markings)):
        mark_dict[markings[i][1]] = vals[i]

    #get only y pixel values for markings
    y_vals = []
    for pt in markings:
        y_vals.append(pt[1])

    #find nearest marker below point
    near = -1
    while near < lvl[1]:
        near = min(y_vals, key=lambda x:abs(x-lvl[1]))
        y_vals.remove(near)

    #estimate remainder
    rem = np.abs(lvl[1] - near)
    cm = (rem/pxl_avg)/100
    est_height = round((mark_dict[near] + cm), 2)
    return est_height

#read in exposed hull image
hull = cv.imread("final\hull_marker.jpg") #path to hull_marker.jpg image file
hull_gray = cv.cvtColor(hull, cv.COLOR_BGR2GRAY)
_,thresh = cv.threshold(hull_gray,200,255,cv.THRESH_BINARY)

#identify hull markings
contours, hier = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

#filter out anything not a marking
good_cnt = []
markings = []
for cnt in contours:
    rect = cv.boundingRect(cnt)
    x,y,w,h = rect
    size = cv.contourArea(cnt)
    if 50 < size < 1500 and 1.2*w > h:
        good_cnt.append(cnt)
        pt = (x,y)
        markings.append(pt)

#display marking coordinates
for pt in markings:
    cv.circle(hull, pt, 1, (0,0,255), 1)

#calculate the average number of pixels between markings
tot_dif = 0
for i in range(0, len(markings)-1):
    dif = markings[i][1] - markings[i+1][1]
    tot_dif = dif + tot_dif

avg_dif = tot_dif/len(markings)

#convert pixels to centimeters (markings on ship are in increments of 10 cm)
p2cm = avg_dif/10

#create and plot imaginary waterline
lvl = (250, 450)
cv.circle(hull, lvl, 1, (255,0,255), 5)

print("the estimated water level at pink point is %.2f meters" %levelEst(markings, lvl, p2cm))

#display hull with marking coordinates
cv.imshow('img', hull)
cv.waitKey()