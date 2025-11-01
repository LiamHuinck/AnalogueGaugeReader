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
    return cv.blur(~mask, (9, 9))

maskedimage = maskgrayframe(grayimage, 0, 200)

# Detect circles
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

for i in circles[0,:]:
    # draw the outer circle
    cv.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(frame,(i[0],i[1]),2,(40,30,200),3)

# Show result
cv.imshow('Detected Circle', frame)
cv.waitKey(0)
cv.destroyAllWindows()