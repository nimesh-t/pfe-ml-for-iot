# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

# Le code camera a ete modifie pour pouvoir faire de l'inference sur la video.

import cv2
################################################
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

import numpy as np
import argparse
import imutils
import os

Class=["Fire","Neutral","Smoke"]
#model = load_model("./Results/MobileNet_tf_1e-5_3_fire_last_model.h5")
model = load_model("./Results/MobileNet_tf_1e-5_3_fire_last_saved_model")


##################################################

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera():
	#prev_frame_time=0
	#new_frame_time=0

	# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
	print(gstreamer_pipeline(flip_method=0))
	cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
	if cap.isOpened():
		window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
	# Window
		while cv2.getWindowProperty("CSI Camera", 0) >= 0:
			ret_val, img = cap.read()            
			#cv2.imshow("CSI Camera", img)
			#new_frame_time=time.time()
			#fps=1/(new_frame_time-prev_frame_time)
			#prev_frame_time=new_frame_time

			#############################################
			cv2.imwrite('temp.jpg',img)
			# Pre traitement de l'image pour qu'il puisse etre passe au modele
			img_proc = tf.keras.preprocessing.image.load_img('temp.jpg',target_size=(224,224))
			img_array=image.img_to_array(img_proc)
			img_array = tf.expand_dims(img_array,0)
			img_array=preprocess_input(img_array)

			#start_time = time.time()
			preds = model.predict(img_array,steps=1)[0]
			#print("inference time : "+ str(time.time()-start_time))

			print(preds)

			idxs = np.argsort(preds)[::-1][:2]

			for (i, j) in enumerate(idxs):
				label = "{}: {:.2f}%".format(Class[j], preds[j] * 100)
				cv2.putText(img, label, (10, (i * 30) + 25), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

			for (label, p) in zip(Class, preds):
				print("{}: {:.2f}%".format(label, p * 100))

			cv2.imshow("Detection de Feu avec MobileNet - CSI Camera",img)

			#############################################

			# This also acts as
			keyCode = cv2.waitKey(30) & 0xFF
			# Stop the program on the ESC key
			if keyCode == 27:
				break
		cap.release()
		cv2.destroyAllWindows()
	else:
		print("Unable to open camera")


if __name__ == "__main__":
    show_camera()
