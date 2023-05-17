import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


################## functions #########################

# resizes an image for easier viewing on screen
def resize_image(img, scale):
    scale_width = int(img.shape[1] * scale)
    scale_height = int(img.shape[0] * scale)
    dim = (scale_width, scale_height)
    resize_img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return resize_img

def levelEst(markings, lvl, pxl_avg):
    mark_dict = {}
    vals = []
    height = 7.5

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
    return y

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

avg_wtl = []
count = 1
frames = []
while ship_vid.isOpened():
    ret, frame = ship_vid.read()
    if ret == True:
        frames.append(count)
        count += 1
        img_warp = cv.warpPerspective(frame, H, (row, col+200))
        height = water_level(img_warp)
        # if height != 0:
            # print("Water level: " + str(height) + " pixels")

        #read in exposed hull image
        # hull = cv.imread("hull_marker.jpg") #path to hull_marker.jpg image file
        hull_gray = cv.cvtColor(img_warp, cv.COLOR_BGR2GRAY)
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
            cv.circle(img_warp, pt, 1, (0,0,255), -1)

        #calculate the average number of pixels between markings
        tot_dif = 0
        for i in range(0, len(markings)-1):
            dif = markings[i][1] - markings[i+1][1]
            tot_dif = dif + tot_dif

        avg_dif = tot_dif/len(markings)

        #convert pixels to centimeters (markings on ship are in increments of 10 cm)
        p2cm = avg_dif/10

        #create and plot imaginary waterline
        lvl = (1100, 450)
        cv.circle(img_warp, lvl, 1, (255,0,255), 5)

        print("The estimated water level at pink point is %.2f meters" %levelEst(markings, lvl, p2cm))

        avg_wtl.append(levelEst(markings, lvl, p2cm))
            
        img_resize = resize_image(img_warp, 0.5)
        cv.imshow('Water Level Detection', img_resize)

        
        key = cv.waitKey(20)
    
        if key == ord('q'):
            break
    else:
        break

avg_wl = sum(avg_wtl)/len(avg_wtl)
print("The average estimated water level at is %.2f meters" %avg_wl)

##---------------Plotting----------------------##
fig = plt.figure()
plt.title('Average Waterline Height vs. Time')
plt.ylabel('Average Waterline (m)')
plt.xlabel('Frame Number')
plt.plot(frames, avg_wtl, 'b-', label = 'Average Frame Waterline')
plt.legend()
plt.show()

