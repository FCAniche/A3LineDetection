import sys
import cv2 as cv
import numpy as np
from moviepy.editor import VideoFileClip
import math

# Convert BGR images to RGB color space
def BGR2RGB(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def RGB2BGR(img):
    return cv.cvtColor(img, cv.COLOR_RGB2BGR)

def mid_point(line):
    for x1,y1,x2,y2 in line:
        midX = (x1 + x2) / 2
        midY = (y1 + y2) / 2
        return np.array([midX, midY])
    
def y_intercept(x,y,m):
    return y - (m*x)

def find_x(y,m,b):
    y = y - b
    return y / m

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    l_mid_points = np.array([0,0])
    l_slope = 0
    l_count = 0
    ly = img.shape[1]
    
    r_mid_points = np.array([0,0])
    r_slope = 0
    r_count = 0
    ry = img.shape[1]
    
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
                    r_mid_points = np.add(r_mid_points, mid)
                    r_slope = r_slope + slope
                    r_count = r_count + 1
                
                    ry = min(ry,y1,y2, round(img.shape[1]/2))
            elif round(slope) != 0:
                l_mid_points = np.add(l_mid_points, mid)
                l_slope = l_slope + slope
                l_count = l_count + 1
                
                ly = min(ly,y1,y2, round(img.shape[1]/2))
    
    # Calculate slope and position of lanes
    if l_count == 0 or r_count == 0:
        return
    
    l_slope = l_slope / l_count
    r_slope = r_slope / r_count
    
    l_mid = np.true_divide(l_mid_points, l_count)
    r_mid = np.true_divide(r_mid_points, r_count)
    
    # Find end points for the lanes
    bottomY = img.shape[1]
    
    l_intercept = y_intercept(l_mid[0], l_mid[1], l_slope)
    r_intercept = y_intercept(r_mid[0], r_mid[1], r_slope)
    
    l_top = round(find_x(ly, l_slope, l_intercept))
    l_bottom = round(find_x(bottomY, l_slope, l_intercept))
    
    r_top = round(find_x(ry, r_slope, r_intercept))
    r_bottom = round(find_x(bottomY, r_slope, r_intercept))
    
    # Draw final lane lines
    cv.line(img, (l_bottom, bottomY), (l_top, ly), [0,255,0], thickness*10)
    cv.line(img, (r_bottom, bottomY), (r_top, ry), [0,255,0], thickness*10)
    
    # Draw mid-point
    cv.circle(img, (round(l_mid[0]), round(l_mid[1])), 10, [0,0,255], thickness*5)
    cv.circle(img, (round(r_mid[0]), round(r_mid[1])), 10, [0,0,255], thickness*5)
    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv.addWeighted(initial_img, α, img, β, γ)

def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
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
    
    # Apply blur
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
    lanes = weighted_img(houghLines, BGR2RGB(img))
    return lanes

# Find lanes in a video by converting to BGR color space first
def FindLanesVid(frame):
    return FindLanes(RGB2BGR(frame))

########################################
def improperUsage():
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
        laneImg = RGB2BGR(FindLanes(img))
        cv.imwrite(sys.argv[3], laneImg)
    else:
        improperUsage()
else:
    improperUsage()