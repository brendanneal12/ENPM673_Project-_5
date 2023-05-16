import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


################## functions #########################

def transformation(image):   
    lower_orange = np.array([0,80,150]) #bgr thresholds
    upper_orange = np.array([80,150,255])
    # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    col,row,_ = np.shape(image)
    x1, y1, x2, y2 = 1130, 0, 1500, 300
    mask = np.zeros((col, row), dtype=np.uint8)
    rect_mask = cv.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    image_rect = cv.bitwise_and(image, image, mask=rect_mask)
    image_filt = cv.GaussianBlur(image_rect,(7,7),0)
    hull_mask = cv.inRange(image_filt, lower_orange, upper_orange)
    hull_filter = cv.bitwise_and(image_filt, image_filt, mask= hull_mask)
    hull_edges = cv.Canny(hull_filter,50,200)
    hull_lines = cv.HoughLinesP(hull_edges,rho=1,theta=np.pi/180,threshold=60,minLineLength=0,maxLineGap=50)
    # print(hull_lines)
    # cv.imshow('Original',frame)
    # cv.imshow('Gray', frame_gray)
    # cv.imshow('Rectangle ROI',image_rect)
    # cv.imshow('Edges',hull_edges)

    ## commented out code can be used to visualize lines on image
    # lines_out = frame.copy()
    # for i in range(len(hull_lines)):
    #     startPoint = (hull_lines[i][0][0], hull_lines[i][0][1])
    #     endPoint = (hull_lines[i][0][2], hull_lines[i][0][3])
    #     cv.line(lines_out, startPoint, endPoint, (255, 0, 0))
    # cv.imshow('Line Output', lines_out)

    p1 = [hull_lines[3][0][0], hull_lines[3][0][1]]
    p2 = [hull_lines[3][0][2], hull_lines[3][0][3]]
    p3 = [hull_lines[4][0][0], hull_lines[4][0][1]]
    p4 = [hull_lines[4][0][2], hull_lines[4][0][3]]

    l1 = [p1[0], 190]
    l2 = [p2[0], 190]
    l3 = [p3[0]-3, 145+1]
    l4 = [p4[0], 145]

    src_pts = np.array([p1,p2,p3,p4])
    dst_pts = np.array([l1,l2,l3,l4])
    # print(src_pts)
    # print(dst_pts)

    H, _ = cv.findHomography(src_pts, dst_pts)
    # print("Homography Matrix:",H)
    return H

def resize_image(img, scale):
    scale_width = int(img.shape[1] * scale)
    scale_height = int(img.shape[0] * scale)
    dim = (scale_width, scale_height)
    resize_img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return resize_img

################### main script ###########################

ship_vid = cv.VideoCapture("hull.mp4")

# while ship_vid.isOpened():
ret, frame = ship_vid.read()
if ret == True:
    col, row, _= np.shape(frame)
    H = transformation(frame)

while ship_vid.isOpened():
    ret, frame = ship_vid.read()
    if ret == True:
        scale = 1
        img_warp = cv.warpPerspective(frame, H, (scale*row, scale*col))
        img_resize = resize_image(img_warp, 0.5)
        cv.imshow('Transformed Image', img_resize)
        
        key = cv.waitKey(20)
    
        if key == ord('q'):
            break
    else:
        break