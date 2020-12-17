import cv2
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import os

import matplotlib.pyplot as plt

model = load_model("./fire_testfull_model.h5")
Class=["Fire","Neutral","Smoke"]

image = cv2.imread("./Inference/test_images/image.jpg")

prediction_datagen = ImageDataGenerator(rescale=1./255)
prediction_generator = prediction_datagen.flow_from_directory(
"Inference/",
target_size=(224, 224),
batch_size=5,
class_mode='categorical',
shuffle=False)

# classify the input image
proba= model.predict_generator(prediction_generator,steps=1)[0]

idxs = np.argsort(proba)[::-1][:2]

# loop over the indexes of the high confidence class labels
for (i, j) in enumerate(idxs):
    # build the label and draw the label on the image
    label = "{}: {:.2f}%".format(Class[j], proba[j] * 100)
    cv2.putText(image, label, (10, (i * 30) + 25), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# show the probabilities for each of the individual labels
for (label, p) in zip(Class, proba):
    print("{}: {:.2f}%".format(label, p * 100))

# show the output image
plt.imshow(image)
plt.show()
cv2.imwrite('output.jpg',image)


"""
# Importations
import tensorflow as tf

#import tensorflow
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

model = keras.models.load_model('fire_testfull_model.h5')
#model.evaluate
#model.summary()
classes = ['Fire','Neutral','Smoke']
model.load_weights('fire_testfull_model.h5')

# Running inference on new data
img = keras.preprocessing.image.load_img(
    "image.jpg", target_size=(224, 224)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]
print(score)
"""
