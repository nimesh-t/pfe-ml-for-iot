import cv2
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

import numpy as np
import argparse
import imutils
import os

import matplotlib.pyplot as plt



# Optimisation par TensorRT
print('Converting to TF-TRT FP16...')
start_time=time.time()
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode='FP16')
converter = trt.TrtGraphConverterV2(
   input_saved_model_dir='./Results/MobileNet_tf_1e-5_3_fire_last_saved_model', conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir='MobileNet_last_TFTRT_FP16')
print("Done Converting to TF-TRT FP16 in "+ str(time.time()-start_time))

# Inference avec le modele optimise
Class=["Fire","Neutral","Smoke"]

print("Chargement du modèle optimise\n")
start_time=time.time()
#model = load_model("./Results/MobileNet_tf_1e-5_3_fire_last_model.h5")
model = load_model("MobileNet_last_TFTRT_FP16")
print("model loading time : "+ str(time.time()-start_time))

print("Chargement de l'image...\n")
img_path='./Inference/test_images/image_3.jpg'
#img_orig = cv2.imread("./Inference/test_images/image.jpg")

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
