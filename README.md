# AnalogueGaugeReader
This repository is for reading out the value of a analogue Gauge, in this specific case a Hygro meter. The idea is that they hygro meter gets read periodically and registers the read values with the datetime time of the reading so that it can be tracked for later analysis.

In this README you will find the different stages/steps of developing the script with the goal described above in mind.

# README Status
Currently the code is further than the README, the README misses the line finding function and the calculation function in its explanation. This will be added some time soon.

# Prerequisites
For following the steps/using this code you need to have certain python packages installed, the required packages can be found in the requirements.txt. To install all of them at once use pip install -r /path/to/requirements.txt .

# Step 1: reading an image
For the reading of images and live webcam feeds as well as the processing we will use OpenCV (https://opencv.org/) this is a package that focusses on providing functions for computer vision solutions.

For early development there is no need to use a livestream constantly, mostly because of convenience, however it will be tested in a later stage with a live camera feed. To do this we will use the opencv function called cv.imread({insert path to image}) with as argument the image we want to use. 

For livestreaming from a webcam make sure to create a capture device using cv.VideoCapture({insert webcam ID or address}) see: https://sandervandevelde.wordpress.com/2024/10/12/recording-rtsp-video-feeds-using-python-and-opencv/ for a detailed guide.

The image below will be used for demonstrative purposes.
![image info](./src/images/20251101_103702.jpg)

# Step 2: Image transformations
The image shown above is pretty big (4000x3000 pixels) since it was taken with a camera. This is overkill for the proposed analysis and can slow the other functions applied to the image. So the image is resized using the function resizeframe, this function uses the image and the width that is wished to rescale the image into the wished for size. To get the original size of the image you can skip this function call. In this README example it gets rescaled to 700 pixels wide. After the rescale we convert the frame into a grayscale image. This will help with finding the gauge, since this is done on contrast. The result is that we have the image below.
![image info](./src/images/gray_rescaled_20251101_103702.jpg)
This image is already easier to process however if we apply the function cv.HoughCircles (which is the OpenCV function that looks for circles in an image) it would find a lot of circles in the image, probably because of differences between values. To make sure we only look at the area of interest we apply a mask using the maskgrayframe function. This takes the grayimage and sets the pixel values between lowerblackvalue & upperblackvalue to 0 (0 means pitchblack 255 means white for pixel values). To make sure there are no stray pixels that can form a circle together we also apply a blur. This helps with reducing the impact of stray pixels that we are not interested in. See the image below for the result after these transformations.
![image info](./src/images/masked_blurred_20251101_103702.jpg)

# Step 3: Detect circles
Using the cv.HoughCircles() command we can find the circles in the image and thus locate our gauge meter. For the command we use the maskedimage and some hyperparameters (these will probably have to be tweaked depending on the use case, look for the documentation from OpenCV for this function for more info). This results in an array with 3 columns containing the center coordinates and the radius of the circle. We draw these points for each of the found circles onto the resized image (with colour) and receive the results shown below.
![image info](./src/images/circle_detected_20251101_103702.jpg)
