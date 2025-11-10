import math
import numpy as np
import cv2 as cv
import json
import time
from datetime import datetime, timezone
import paho.mqtt.publish as publish
from credentials import login, ip

broker_address = "localhost"
broker_port = 1883

topic = "test/topic1"
hygro_reading = 0
# Helper functions
def resizeframe(frame, new_width):
    # Get the original dimensions
    original_height, original_width = frame.shape[:2]
    # Define new width while maintaining the aspect ratio
    aspect_ratio = new_width / original_width
    new_height = int(original_height * aspect_ratio)  # Compute height based on aspect ratio
    return cv.resize(frame, (new_width, new_height))

def maskgrayframe(maskingframe, lowerblackvalue, upperblackvalue):
    lower_black, upper_black = np.array([lowerblackvalue]), np.array([upperblackvalue])
    mask = cv.inRange(maskingframe, lower_black, upper_black)
    return cv.blur(~mask, (5, 5))

def calculateangle(lines,circles, correctiondegrees):
    circlex = circles[0][0][0]
    circley = circles[0][0][1]
    x1 = lines[0][0][0]
    y1 = lines[0][0][1]
    x2 = lines[0][0][2]
    y2 = lines[0][0][3]

    distancecircle1 = math.sqrt((x1-circlex)**2+ (y1-circley)**2)
    distancecircle2 = math.sqrt((x2-circlex)**2+ (y2-circley)**2)
    if distancecircle1 >= distancecircle2:
        angle = math.degrees(math.atan2(y1-circley, x1-circlex))
    else:
        angle = math.degrees(math.atan2(y2-circley, x2-circlex))
    totalangle = correctiondegrees+angle
    return totalangle

capturedevice = cv.VideoCapture('rtsp://'+login+'@'+ip+'/stream1')
if not capturedevice.isOpened():
    print("Can't open camera")
    exit()

count = 0

while True:
    ret, frame = capturedevice.read()

    frame = resizeframe(frame, 700)
    
    # convert image to grayscale
    grayimage = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    maskedimage = maskgrayframe(grayimage, 0, 90)

    circles = cv.HoughCircles(
        maskedimage,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=400,      
        param1=40,         
        param2=80,
        minRadius=50,       
        maxRadius=95
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

    outlinedimage = cv.Canny(maskedimage, 100, 300, None, 5)    
    lines = cv.HoughLinesP(outlinedimage, 1, np.pi / 180, 60, None, minLineLength = 60, maxLineGap= 10)
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

    # calculation of the angle that the line is drawn at and which side it is pointing to
    # call function to calculate the angle required to calculate the hygro value
    # the value of 140 is for the offset of the 0 value (should be on the Y value of the circle center on the right hand side) in degrees
    try:
        totalangle = calculateangle(lines,circles, 140)
    except:
        pass

    # calculation of the hygro reading
    # 270 is the measuring degrees of the hygro meter
    try:
        hygro_reading = (totalangle)/(270)*100
        frame = cv.putText(frame, f"Humidity: {hygro_reading:.0f}%", (00, 200), cv.FONT_HERSHEY_SIMPLEX, 
                    1, (255,255,255), 2, cv.LINE_AA)
        if count == 0 or count % 30 == 0:
            datadictionary = {"key": "humidity", "value":int(hygro_reading), "timestamp":datetime.now(timezone.utc), "deviceId":"visiontest"}
            payload = json.dumps(datadictionary, default=str)
            publish.single(topic, payload, hostname=broker_address, port= broker_port)
    except:
        print("No reading possible")

    # Show result
    cv.imshow('Detected Circle', frame)

    count += 1
    # press q to stop the capturing
    if cv.waitKey(1) == ord('q'):
        break

# release the capture device for other programs to use
capturedevice.release()
cv.destroyAllWindows()