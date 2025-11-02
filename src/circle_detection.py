import math
import numpy as np
import cv2 as cv

# Read the first frame to confirm capturing
frame = cv.imread('src/images/20251101_103702.jpg')

def resizeframe(frame, new_width):
    # Get the original dimensions
    original_height, original_width = frame.shape[:2]
    # Define new width while maintaining the aspect ratio
    aspect_ratio = new_width / original_width
    new_height = int(original_height * aspect_ratio)  # Compute height based on aspect ratio
    return cv.resize(frame, (new_width, new_height))

frame = resizeframe(frame, 700)

# convert image to grayscale
grayimage = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

def maskgrayframe(maskingframe, lowerblackvalue, upperblackvalue):
    lower_black, upper_black = np.array([lowerblackvalue]), np.array([upperblackvalue])
    mask = cv.inRange(maskingframe, lower_black, upper_black)
    blank = np.zeros((747, 714), dtype='uint8')
    return cv.blur(~mask, (10, 10))

maskedimage = maskgrayframe(grayimage, 0, 200)

circles = cv.HoughCircles(
    maskedimage,
    cv.HOUGH_GRADIENT,
    dp=1,
    minDist=400,      
    param1=50,         
    param2=100,       
    minRadius=100,       
    maxRadius=0
)

circles = np.uint16(np.around(circles))
if circles is not None:
    for i in circles[0,:]:
        #draw outer circles
        cv.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
        #draw inner circles
        cv.circle(frame,(i[0],i[1]),2,(40,30,200),3)

outlinedimage = cv.Canny(maskedimage, 50, 200, None, 3)    
lines = cv.HoughLines(outlinedimage, 1, np.pi / 180, 80, None, 0, 0)

if lines is not None:
    if len(lines) > 1:
        print("multiple lines found no analysis to be done")
    else:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + (-626+100)*(-b)), int(y0 + (-626+100)*(a)))
            pt2 = (int(x0 - (626+100)*(-b)), int(y0 - (626+100)*(a)))
            cv.line(frame, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
else:
    print("no lines found")
#calculation of the hygro reading
# where the 0.17 is a correction for the rotation of the meter
# 1.5 pi is the radian of the meter
hygro_reading = (theta+0.17*1.5*math.pi)/(1.5*math.pi)
print(hygro_reading)

# Show result
cv.imshow('Detected Circle', frame)
cv.waitKey(0)
cv.destroyAllWindows()