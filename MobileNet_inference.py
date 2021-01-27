import cv2
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

import matplotlib.pyplot as plt


print("Debut du programme d'inférence...\n")

Class=["Fire","Neutral","Smoke"]

print("Chargement du modèle\n")
start_time=time.time()
#model = load_model("./Results/MobileNet_tf_1e-5_3_fire_last_model.h5")
model = load_model("./Results/MobileNet_tf_1e-5_3_fire_last_saved_model")
print("model loading time : "+ str(time.time()-start_time))

print("Chargement de l'image...\n")
img_path='./Inference/test_images/image_3.jpg'
img_orig = cv2.imread("./Inference/test_images/image.jpg")

# Pre traitement de l'image pour qu'il puisse etre passe au modele
img = image.load_img(img_path,target_size=(224,224))
img_array=image.img_to_array(img)
img_array = tf.expand_dims(img_array,0)
img_array=preprocess_input(img_array)
#img_array = tf.constant(img_array)

start_time = time.time()
preds = model.predict(img_array,steps=1)[0]
print("inference time : "+ str(time.time()-start_time))

print(preds)
"""
idxs = np.argsort(preds)[::-1][:2]

# loop over the indexes of the high confidence class labels
for (i, j) in enumerate(idxs):
    # build the label and draw the label on the image
    label = "{}: {:.2f}%".format(Class[j], preds[j] * 100)
    cv2.putText(img_orig, label, (10, (i * 30) + 25), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# show the probabilities for each of the individual labels
for (label, p) in zip(Class, preds):
    print("{}: {:.2f}%".format(label, p * 100))

# show the output image
plt.imshow(img_orig)
plt.show()
cv2.imwrite('test.jpg',img_orig)"""
