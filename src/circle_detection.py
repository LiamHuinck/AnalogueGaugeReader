import math
import numpy as np
import cv2 as cv

# Read the first frame to confirm capturing
frame = cv.imread('src/images/20251102_142248.jpg')

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
    return cv.blur(~mask, (7, 7))

maskedimage = maskgrayframe(grayimage, 0, 130)

circles = cv.HoughCircles(
    maskedimage,
    cv.HOUGH_GRADIENT,
    dp=1,
    minDist=400,      
    param1=50,         
    param2=100,       
    minRadius=100,       
    maxRadius=180
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        #draw outer circles
        cv.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
        #draw inner circles
        cv.circle(frame,(i[0],i[1]),2,(40,30,200),3)
        mask = np.zeros(maskedimage.shape[:2], dtype="uint8")
        cv.circle(mask, (i[0],i[1]), i[2], 255, -1)
        maskedimage = cv.bitwise_and(maskedimage, maskedimage, mask=mask)

outlinedimage = cv.Canny(maskedimage, 40, 100, None, 3)    
lines = cv.HoughLinesP(outlinedimage, 1, np.pi / 90, 70, None, minLineLength = 110, maxLineGap= 8)
if lines is not None:
    if len(lines) > 1:
        print("multiple lines found")
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    else:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
else:
    print("no lines found")
#calculation of the hygro reading
# where the 0.17 is a correction for the rotation of the meter
# 1.5 pi is the radian of the meter


try:
    hygro_reading = (1+0.17*1.5*math.pi)/(1.5*math.pi)
    print(hygro_reading)
except:
    print("No reading possible")

# Show result
cv.imshow('Detected Circle', frame)
cv.waitKey(0)
cv.destroyAllWindows()