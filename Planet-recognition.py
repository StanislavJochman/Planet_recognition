# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import subprocess

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the RGB color space, then initialize the
# list of tracked points
LowerRed = (100,0,0)
UpperRed = (180,100,130)
LowerBlue = (0,0,80)
UpperBlue = (20,80,255)
LowerYellow = (100,100,0)
UpperYellow = (230,200,90)
LowerGreen = (0,80,60)
UpperGreen = (30,255,180)
LowerPurple = (30,0,60)
UpperPurple = (120,50,130)


pts = deque(maxlen=args["buffer"])


if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# keep looping
while True:
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame,height=680)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	maskr = cv2.inRange(rgb, LowerRed, UpperRed)
	maskr = cv2.erode(maskr, None, iterations=2)
	maskr = cv2.dilate(maskr, None, iterations=2)
	maskb = cv2.inRange(rgb, LowerBlue, UpperBlue)
	maskb = cv2.erode(maskb, None, iterations=2)
	maskb = cv2.dilate(maskb, None, iterations=2)
	masky = cv2.inRange(rgb, LowerYellow, UpperYellow)
	masky = cv2.erode(masky, None, iterations=2)
	masky = cv2.dilate(masky, None, iterations=2)
	maskg = cv2.inRange(rgb, LowerGreen, UpperGreen)
	maskg = cv2.erode(maskg, None, iterations=2)
	maskg = cv2.dilate(maskg, None, iterations=2)
	maskp = cv2.inRange(rgb, LowerPurple, UpperPurple)
	maskp = cv2.erode(maskp, None, iterations=2)
	maskp = cv2.dilate(maskp, None, iterations=2)



	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cntsr = cv2.findContours(maskr.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cntsr = cntsr[0] if imutils.is_cv2() else cntsr[1]
	center = None

	cntsb = cv2.findContours(maskb.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cntsb = cntsb[0] if imutils.is_cv2() else cntsb[1]
	center = None

	cntsy = cv2.findContours(masky.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cntsy = cntsy[0] if imutils.is_cv2() else cntsy[1]
	center = None

	cntsg = cv2.findContours(maskg.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cntsg = cntsg[0] if imutils.is_cv2() else cntsg[1]
	center = None

	cntsp = cv2.findContours(maskp.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cntsp = cntsp[0] if imutils.is_cv2() else cntsp[1]
	center = None


	# only proceed if at least one contour was found
	if len(cntsr) > 15:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cntsr, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		center_x = center[0]
		center_y = center[1]
		cv2.putText(frame,'Venusa',(int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255),2,cv2.LINE_AA)
		cv2.circle(frame, (int(x), int(y)), int(radius),(0, 0, 255), 2)
		cv2.circle(maskr, (int(x), int(y)), int(radius)+30,(60, 60, 60), 2)


	if len(cntsb) > 15:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cntsb, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		center_x = center[0]
		center_y = center[1]
		cv2.putText(frame,'Uran',(int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2,cv2.LINE_AA)
		cv2.circle(frame, (int(x), int(y)), int(radius),(255, 0, 0), 2)
	if len(cntsy) > 15:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cntsy, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		center_x = center[0]
		center_y = center[1]
		cv2.putText(frame,'Slnko',(int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 225, 255),2,cv2.LINE_AA)
		cv2.circle(frame, (int(x), int(y)), int(radius),(0, 225, 255), 2)

	if len(cntsg) > 10:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cntsg, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		center_x = center[0]
		center_y = center[1]
		cv2.putText(frame,'Zem',(int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 225,0),2,cv2.LINE_AA)
		cv2.circle(frame, (int(x), int(y)), int(radius),(0, 225,0), 2)
	if len(cntsp) > 15:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cntsp, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		center_x = center[0]
		center_y = center[1]
		cv2.putText(frame,'Neptun',(int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,(91, 0, 239),2,cv2.LINE_AA)
		cv2.circle(frame, (int(x), int(y)), int(radius),(91, 0, 239), 2)

	cv2.rectangle(frame, (0, 50), (1200, 0), (0, 0, 0), -1)
	cv2.putText(frame,'Priloz lopticku a over si aku predstavuje planetu',(100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(230, 230, 230),2,cv2.LINE_AA)
	cv2.rectangle(frame, (0, 700), (1200, 630), (0, 0, 0), -1)
	cv2.putText(frame,'Created by Stanislav Jochman',(10, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(230, 230, 230),1,cv2.LINE_AA)
	cv2.putText(frame,'www.stanislavjochman.wz.sk',(630, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(230, 230, 230),1,cv2.LINE_AA)
	# show the frame to our screen
	cv2.imshow("Frame", frame)
	#cv2.imshow("Masky", maskr)
	key = cv2.waitKey(1) & 0xFF
	# if the 'esc' key is pressed, stop the loop
	if key == 27:
		break

#release camera
vs.stop()
# close all windows
cv2.destroyAllWindows()
