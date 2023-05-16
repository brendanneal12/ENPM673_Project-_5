import numpy as np
import cv2 as cv
import pytesseract
import re

# Downloaded pytesseract and put the location here
pytesseract.pytesseract.tesseract_cmd = "C:/Users/harjo/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"

################## functions #########################

# resizes an image for easier viewing on screen
def resize_image(img, scale):
    scale_width = int(img.shape[1] * scale)
    scale_height = int(img.shape[0] * scale)
    dim = (scale_width, scale_height)
    resize_img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return resize_img

# Estimates the water level and returns the height in pixels
def water_level(frame):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, thresh_frame = cv.threshold(gray_frame, 215, 255, cv.THRESH_BINARY)

    mask = back_subtract.apply(thresh_frame) # separate the background and foreground

    kernel = np.ones((3, 3), np.uint8)
    morph_frame = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations = 1)

    contours, hierarchy = cv.findContours(morph_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    y = 0
    if len(contours) > 0:
        cont_max = max(contours, key = cv.contourArea)  # get the contour with the biggest area
        x, y, w, h = cv.boundingRect(cont_max)
        #cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv.line(frame, (x, y), (x+150, y), (0, 255, 255), 3)  # draw horizontal line at the water height level

    #cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
    # cv.imshow('Water Level', frame)
    return y, thresh_frame

# computes the homography matrix to tranform the frames so that draft marks are horizontal
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

    ## commented out code below can be used to visualize lines on image
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

    l1 = [p1[0]+100, 190]
    l2 = [p2[0]+100, 190]
    l3 = [p3[0]+97, 145+1]
    l4 = [p4[0]+100, 145]

    src_pts = np.array([p1,p2,p3,p4])
    dst_pts = np.array([l1,l2,l3,l4])
    # print(src_pts)
    # print(dst_pts)

    H, _ = cv.findHomography(src_pts, dst_pts)
    # print("Homography Matrix:",H)
    return H



################### main script ###########################

ship_vid = cv.VideoCapture("hull.mp4")

# while ship_vid.isOpened():
ret, frame = ship_vid.read()
if ret == True:
    col, row, _= np.shape(frame)
    H = transformation(frame)

back_subtract = cv.createBackgroundSubtractorMOG2()

frame_count = 0
total_frame = 0
level_list = []

while ship_vid.isOpened():
    ret, frame = ship_vid.read()

    if ret == True:
        img_warp = cv.warpPerspective(frame, H, (row, col+200))
        height, thresh_frame = water_level(img_warp)

        sharp_kernel = np.array([[-1, -1, -1], [-1, 21, -1], [-1, -1, -1]])     
        sharp_warped = cv.filter2D(thresh_frame, -1, sharp_kernel)
        warp_inverted = 255-sharp_warped

        digits = pytesseract.image_to_string(warp_inverted, lang='eng', config='--psm 11 --oem 3 -c tessedit_char_whitelist=0124689M')

        # for x, y coordinates of bounding rectangles on characters found
        #box = pytesseract.image_to_boxes(warp_inverted, lang='eng', config='--psm 11 --oem 3 -c tessedit_char_whitelist=0124689M')
        #print(box)  # compare lowest character y value with waterline estimate

        M_vals = re.findall("\d+M", digits)         # one or more digits followed by a M

        digit_vals = re.findall("\w+", digits)  # convert the strings found into a list

        print(digit_vals)

        if len(M_vals) >= 1:
            M_digit = M_vals[-1]                        # get the lower M value detected
            first_digit = int(M_digit.replace('M',''))  # replace M with empty space and convert to int
            first_digit = first_digit-1
            
            last_digit = digit_vals[-1]
            if 'M' in last_digit:   # if the lowest digit has M in it, the tenths place is zero
                last_digit = 0
                first_digit = first_digit+1
            
            digit_combined = str(first_digit) + "." + str(last_digit)
            level = float(digit_combined)
            print(level)
            frame_count += 1
            level_list.append(level)

            # store previous value, if change is more than 3 between values, dont use the value (outlier)
            
        else:
            print("Height not found")

        total_frame += 1

        cv.imshow('Frame', resize_image(frame, 0.5))
        #cv.waitKey(1000)
        #print("end")

        if cv.waitKey(25) & 0xFF == ord('e'):
            break
    else:
        break

avg_level = sum(level_list)/len(level_list)
print(avg_level)
print("Frame number: " + str(frame_count))
print("Total Frame number: " + str(total_frame))
ship_vid.release()
cv.destroyAllWindows()