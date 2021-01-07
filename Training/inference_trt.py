import os
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input


class_names = ["Fire","Neutral","Smoke"]

def predict_tftrt(input_saved_model):
    #Runs prediction on a single image and shows the result.
    #input_saved_model (string): Name of the input model stored in the current dir
    
    img_path = './Inference/test_images/image.jpg' 
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = tf.constant(x)
    

    saved_model_loaded = tf.saved_model.load(input_saved_model)

    # Les signatures contiennent des paires <cle,valeur> avec les definitions du graph
    # Par defaut, si rien n'a ete specifie lors de model.save(), 'serving_default' est utilise
    signature_keys = list(saved_model_loaded.signatures.keys())
    print(signature_keys)

    infer = saved_model_loaded.signatures['serving_default']
    print(infer.structured_outputs)

    labeling = infer(x) # eg { 'dense' : <127,3,3,1  , [[proba][proba]], ...>

    preds = labeling['dense'].numpy()[0]
    print(preds)


# Optimisation par TensorRT
# L'utilisation de INT8 requiert une calibration au prealable avec une BDD de calibration
print('Converting to TF-TRT FP16...')
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode='FP16')
converter = trt.TrtGraphConverterV2(
   input_saved_model_dir='fire_MobileNet_3_fire_full', conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir='fire_MobileNet_3_fire_full_TFTRT_FP16')
print('Done Converting to TF-TRT FP16')

# Inference avec le modele optimise
predict_tftrt('fire_MobileNet_3_fire_full_TFTRT_FP16')




