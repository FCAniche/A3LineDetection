import sys
import cv2 as cv
import numpy as np
from moviepy.editor import VideoFileClip
import math

# Convert BGR images to RGB color space
def bgr_2_rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def rgb_2_bgr(img):
    return cv.cvtColor(img, cv.COLOR_RGB2BGR)

def mid_point(line):
    for x1,y1,x2,y2 in line:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        return np.array([mid_x, mid_y])
    
def y_intercept(x, y, m):
    return y - (m * x)

def find_x(y, m, b):
    y = y - b
    return y / m

def draw_lines(img, lines, thickness=2):
    left_mid_points = np.array([0,0])
    left_slope = 0
    left_count = 0
    left_y = img.shape[1]
    
    right_mid_points = np.array([0,0])
    right_slope = 0
    right_count = 0
    right_y = img.shape[1]
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            # Find the slope
            slope = (y2-y1)/(x2-x1)
            mid = mid_point(line)
            
            if math.isinf(slope):
                continue
                            
            # Group all lines into left and right
            if slope > 0:
                if round(slope) != 0:
                    right_mid_points = np.add(right_mid_points, mid)
                    right_slope += slope
                    right_count += 1
                
                    right_y = min(right_y, y1, y2, round(img.shape[1] / 2))
            elif round(slope) != 0:
                left_mid_points = np.add(left_mid_points, mid)
                left_slope += slope
                left_count += 1
                
                left_y = min(left_y, y1, y2, round(img.shape[1] / 2))
    
    # Calculate slope and position of lanes
    if left_count == 0 or right_count == 0:
        return
    
    left_slope /= left_count
    right_slope /= right_count
    
    left_mid = np.true_divide(left_mid_points, left_count)
    right_mid = np.true_divide(right_mid_points, right_count)
    
    # Find end points for the lanes
    bottom_Y = img.shape[1]
    
    left_intercept = y_intercept(left_mid[0], left_mid[1], left_slope)
    right_intercept = y_intercept(right_mid[0], right_mid[1], right_slope)
    
    left_top = round(find_x(left_y, left_slope, left_intercept))
    left_bottom = round(find_x(bottom_Y, left_slope, left_intercept))
    
    right_top = round(find_x(right_y, right_slope, right_intercept))
    right_bottom = round(find_x(bottom_Y, right_slope, right_intercept))
    
    # Draw final lane lines
    cv.line(img, (left_bottom, bottom_Y), (left_top, left_y), [0,255,0], thickness * 10)
    cv.line(img, (right_bottom, bottom_Y), (right_top, right_y), [0,255,0], thickness * 10)
    
    # Draw mid-point
    cv.circle(img, (round(left_mid[0]), round(left_mid[1])), 10, [0,0,255], thickness * 5)
    cv.circle(img, (round(right_mid[0]), round(right_mid[1])), 10, [0,0,255], thickness * 5)
    
def hough_lines(img, rho, theta, threshold, min_line_length, max_line_gap):
    lines = cv.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    return cv.addWeighted(initial_img, alpha, img, beta, gamma)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning an image only where mask pixels are not zero
    masked_image = cv.bitwise_and(img, mask)
    return masked_image

# Split an image into seperate BGR channels
def Split(img):
    b, g, r = cv.split(img)
    return b, g, r

def GaussianBlur(img, kernel):
    return cv.GaussianBlur(img, (kernel, kernel), 0)

# Simple Canny Edge Detection Function
def Canny(img, minVal, maxVal):
    return cv.Canny(img, minVal, maxVal)

def FindLanes(img):
    # Split channels
    blue,green,red = Split(img)
    
    # Apply Gaussian Blur
    blurred = GaussianBlur(red, 15)
    
    # Apply Canny Edge Detection to the processed image
    cannyEdges = Canny(blurred, 10, 150)

    # Crop image to show just the road
    shape = cannyEdges.shape
    imgVertices = np.array([[(shape[1] - (shape[1]/2.3),shape[0]/1.8),(shape[1]/2.3,shape[0]/1.8),(0,shape[0]),(shape[1],shape[0])]],dtype=np.int32)
    regionEdges = region_of_interest(cannyEdges,imgVertices)
    
    # Find hough lines in edges
    houghLines = hough_lines(regionEdges,2,np.pi/180,25,1,1.75)

    # Overlay lines on initial image
    lanes = weighted_img(houghLines, bgr_2_rgb(img))
    return lanes

# Find lanes in a video by converting to BGR color space first
def FindLanesVid(frame):
    return FindLanes(rgb_2_bgr(frame))

########################################
def improper_usage():
    print("Usage: python3 laneDetection.py file_type(either image or video) file_path output_path")
    exit

if len(sys.argv) == 4:
    ftype = sys.argv[1]
    if ftype == 'video':
        # Write video
        vid = VideoFileClip(sys.argv[2])
        laneVid = vid.fl_image(FindLanesVid)
        laneVid.write_videofile(sys.argv[3], audio=False)
    elif ftype == 'image':
        # Write Image
        img = cv.imread(sys.argv[2])
        laneImg = rgb_2_bgr(FindLanes(img))
        cv.imwrite(sys.argv[3], laneImg)
    else:
        improper_usage()
else:
    improper_usage()
