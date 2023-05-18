import numpy as np
import cv2 as cv
import pytesseract
import re
from matplotlib import pyplot as plt

# Downloaded pytesseract and put the location here
#pytesseract.pytesseract.tesseract_cmd = "C:/Users/harjo/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

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
        #cv.line(frame, (x, y), (x+150, y), (0, 255, 255), 3)  # draw horizontal line at the water height level

    #cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
    # cv.imshow('Water Level', frame)
    # cv.waitKey(10)
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


def levelEst(markings, lvl, pxl_avg, y_info):
    mark_dict = {}
    vals = []
    he = y_info[0] # found either 9 or 10

    # find where the dash mark closest to the bottom of this y value
    y_vals = []
    diff = []
    for pt in markings:
        y_vals.append(pt[1])
        diff_y = abs(pt[1] - y_info[2])
        diff.append(diff_y)

    # find the index of the value closest to 9 or 10 meters in the list
    notc = diff.index(min(diff))

    #creating marking dictionary for pixel to meter translation
    for i in range(0, len(markings)):
        height = round((he - 0.1*(i - notc)), 2)
        vals.append(height)
    vals.reverse()
    for i in range(0, len(markings)):
        mark_dict[markings[i][1]] = vals[i]

    #find nearest marker below point
    near = -1
    if lvl > markings[0][1]:
        lvl = markings[0][1] - 1
    while near < lvl:
        near = min(y_vals, key=lambda x:abs(x-lvl))
        y_vals.remove(near)

    #estimate remainder
    rem = np.abs(lvl - near)
    cm = (rem/pxl_avg)/100
    est_height = round((mark_dict[near] + cm), 2)
    return est_height


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
count = 1
frames = []
maximum_height = 0

video_name = ('group9_final') #Initialize Video Object
fourcc = cv.VideoWriter_fourcc(*"mp4v") #Initialize Video Writer using fourcc
video = cv.VideoWriter(str(video_name)+".mp4", fourcc, 30,(960,640)) #Initialize the Name, method, frame rate, and size of Video.

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
        box = pytesseract.image_to_boxes(warp_inverted, lang='eng', config='--psm 11 --oem 3 -c tessedit_char_whitelist=0124689M')

        digit_vals = re.findall("\w+", digits)  # convert the strings found into a list
        M_vals = re.findall("\d+M", digits)         # one or more digits followed by a M

        # find the location of the letter M found in the box output
        m_loc = []
        for i in range(len(box)):
            if box[i] == 'M':
                m_loc.append(i)
        
                  
        # use the gray hull to find contours and warp the image
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
            if 50 < size < 900 and 0.8*w > h and x > 750 and x < 1150:
                good_cnt.append(cnt)
                pt = (x,y)
                markings.append(pt)

        #display marking coordinates
        #for pt in markings:
            #cv.circle(img_warp, pt, 1, (0,0,255), -1)

        #calculate the average number of pixels between markings
        tot_dif = 0
        for i in range(0, len(markings)-1):
            dif = markings[i][1] - markings[i+1][1]
            tot_dif = dif + tot_dif

        avg_dif = tot_dif/len(markings)

        #convert pixels to centimeters (markings on ship are in increments of 10 cm)
        p2cm = avg_dif/10

        if m_loc and M_vals:
            find_y = []
            # record 10 meters as the marker found with the bottom left corner x and y values
            if M_vals[0] == '10M':
                idx = m_loc[0]
                find_y.append(10)
                find_y.append(float(box[(idx+2):(idx+6)]))
                find_y.append(float(box[(idx+7):(idx+11)]))
            # record 9 meters as the marker found with the bottom left corner x and y values
            elif digit_vals[-1] == '9M':
                idx = m_loc[-1]
                find_y.append(9)
                find_y.append(float(box[(idx+2):(idx+6)]))   # x value of the bottom left corner
                find_y.append(float(box[(idx+7):(idx+11)]))  # y value of the bottom left corner
                lvl = (int(x), int(y))
            #cv.circle(img_warp, lvl, 1, (255,0,255), 5)
            if find_y:
                level = levelEst(markings, height, p2cm, find_y)
                print("The estimated water level is %.2f meters" %level)

                level_list.append(level)
                frames.append(count)
                count += 1
        
        if len(M_vals) >= 1:    # M value is found
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

           level_list.append(level)
           frames.append(count)
           count += 1

        total_frame += 1
                
        img_resize = resize_image(img_warp, 0.5)
        water_level_text = "Water level : "+ str(level)

        cv.putText(img_resize, water_level_text, (30,30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)
        cv.imshow('Water Level Detection', img_resize)
        video.write(img_resize) #Write to Video
        # cv.waitKey(20)
        # print("end")

        if cv.waitKey(25) & 0xFF == ord('e'):
           break
    else:
        break

# Post-processing to remove outliers
levels = np.array(level_list)
mean = np.mean(levels)
std_dev = np.std(levels)
lower = mean-2*std_dev
upper = mean+2*std_dev

final_list = []
for lvl in level_list:
    if lvl < lower or lvl > upper:
        frames.pop()
    else:
        final_list.append(lvl)


#print(final_list)
avg_level = sum(final_list)/len(final_list)
print("Average water level on hull: %.2f meters" %avg_level)

##----------Extend Video a Little Longer to see Everything--------##
#for i in range(200):
#    video.write(cv.cvtColor(img_resize, cv.COLOR_RGB2BGR))

video.release()
ship_vid.release()
cv.destroyAllWindows()


##========Plotting========##
fig = plt.figure()
plt.title('Average Waterline Height vs. Time')
plt.ylabel('Average Waterline (m)')
plt.xlabel('Frame Number')
plt.scatter(frames, final_list, color = 'b', label = 'Average Frame Waterline') # plot the points
plt.ylim(6, 12)      # scale the y-axis
plt.legend()
plt.show()