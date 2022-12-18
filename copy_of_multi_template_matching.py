

### Import Packages
# import the necessary packages
from imutils.object_detection import non_max_suppression
from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2

"""### Function to display images """

def plt_imshow(title, image):
	# convert the image frame BGR to RGB color space and display it
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.imshow(image)
	plt.title(title)
	plt.grid(False)
	plt.show()

"""### Implementing multi-template matching with OpenCV"""

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", type=str, required=True,
#	help="path to input image where we'll apply template matching")
#ap.add_argument("-t", "--template", type=str, required=True,
#	help="path to template image")
#ap.add_argument("-b", "--threshold", type=float, default=0.8,
#	help="threshold for multi-template matching")
#args = vars(ap.parse_args())

# since we are using Jupyter Notebooks we can replace our argument
# parsing code with *hard coded* arguments and values
args = {
	"image": "images/fruits_image.jpg",
    "template": "images/apple.png",
    "threshold": 0.8
}

# load the input image and template image from disk, then grab the
# template image spatial dimensions
print("[INFO] loading images...")
image = cv2.imread(args["image"])
template = cv2.imread(args["template"])
(tH, tW) = template.shape[:2]

# display the  image and template to our screen
plt_imshow("Image", image)
plt_imshow("Template", template)

# convert both the image and template to grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# perform template matching
print("[INFO] performing template matching...")
result = cv2.matchTemplate(imageGray, templateGray,
	cv2.TM_CCOEFF_NORMED)

# find all locations in the result map where the matched value is
# greater than the threshold, then clone our original image so we
# can draw on it
(yCoords, xCoords) = np.where(result >= args["threshold"])
clone = image.copy()
print("[INFO] {} matched locations *before* NMS".format(len(yCoords)))

# loop over our starting (x, y)-coordinates
for (x, y) in zip(xCoords, yCoords):
	# draw the bounding box on the image
	cv2.rectangle(clone, (x, y), (x + tW, y + tH),
		(255, 0, 0), 3)

# show our output image *before* applying non-maxima suppression
plt_imshow("Before NMS", clone)

# initialize our list of rectangles
rects = []

# loop over the starting (x, y)-coordinates again
for (x, y) in zip(xCoords, yCoords):
	# update our list of rectangles
	rects.append((x, y, x + tW, y + tH))

# apply non-maxima suppression to the rectangles
pick = non_max_suppression(np.array(rects))
print("[INFO] {} matched locations *after* NMS".format(len(pick)))

# loop over the final bounding boxes
for (startX, startY, endX, endY) in pick:
	# draw the bounding box on the image
	cv2.rectangle(image, (startX, startY), (endX, endY),
		(255, 0, 0), 3)

# show the output image
plt_imshow("After NMS", image)
