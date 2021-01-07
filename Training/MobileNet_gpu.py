"""
Exécution : python MobileNet.py
Sorties :  .model (trained model), .png (graphes avec les précisions en fonction du nb d'epochs),
.txt (matrice de confusion)
"""
# Importations

import keras
from keras.models import load_model
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import backend as K

import tensorflow as tf

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np

import os
import sys

"""
# N'utilise que l'espace requis

print("Debut controle gpu\n\n")

# tensorflow.org/guide/gpu

# Methode 1 - toujours la meme erreur (cannot allocate memory)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# Methode 2 - alloue une quantite fixe de memoire => crash/freeze au delà de 1 GB, ne résoud pas le problème
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate @memory_limit MB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2500)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

print("Fin controle gpu\n\n")
"""


# Paramètres à modifier : nbclass, epochs(mettre à 1 pour tester si le modèle tourne), bs (batch size)
# chemins (relatifs/absolus) vers les 2 BDDs (ce script suppose que les données sont déjà séparés (training+validation)
NBCLASS= 3
EPOCHS = 20
BS = 5
BDD = "fire"
train_dir = './Dataset/Train'
val_dir = './Dataset/Test'
IMAGE_DIMS = (224, 224, 3)

# SAVE_NAME génère le nom du fichier de sauvegarde :
# nomdecescript+nbclass+bdd+last/full(plustard) en fonction de training from scratch ou pas
# eg MobileNet_3_fire_full
SAVE_NAME = os.path.basename(sys.argv[0][::-1][3:][::-1])+"_"+str(NBCLASS)+"_"+BDD+"_"

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
	#rotation_range=25,
	#width_shift_range=0.1,
	#height_shift_range=0.1,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True,
	fill_mode="nearest")
validation_datagen = ImageDataGenerator(rescale=1./255)

# Import data
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=BS,
        class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=BS,
        class_mode='categorical',
        shuffle=False
        )


# MobileNet
print("Chargement du modèle de base...")

# On charge un modèle existants avec des poids de imagenet sans la dernière couche(top)
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output

# On rajoute une couche 'Pooling' et ensuite, la dernière couche (avec comme fonction d'activation 'softmax' car multi-class+single label)
# qui renvoit le vecteur avec les probabilités de nos classes
x = GlobalAveragePooling2D()(x)
predictions = Dense(NBCLASS, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Training attribute of base_model layers
# On choisit si on veut réentrainer les couches du modèle de base (True par défaut, spécifié False sinon)
for layer in base_model.layers:
    layer.trainable = True
    trainable_str = "last" if layer.trainable==False else "full"
SAVE_NAME += trainable_str
model.summary()

# Définition des hyperparam pour que le modèle sache comment converger vers les poids optimaux
# 'categorical_crossentropy' car multi-class+single label
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])


# Define callbacks
# Les callbacks sont des instructions au modèle qu s'éxécutent pendant l'entraînement

# ModelCheckpoint permet de sauvegarder systématiquement après chaque epoch le modèle
# mc = ModelCheckpoint('best_fire_'+SAVE_NAME+'.model', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# EarlyStopping arrête le modèle s'il ne s'est pas amélioré de (min_delta) après (patience) epochs
# Il sauvegarde les meilleurs poids.
es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.02,patience=10,restore_best_weights=True)

# Entrainement
# [ToDo] Remplace model.fit_generator par model.fit
H = model.fit_generator(steps_per_epoch=train_generator.n//train_generator.batch_size,
        generator=train_generator,
        epochs=EPOCHS,
		validation_data=validation_generator,
        validation_steps=validation_generator.n//validation_generator.batch_size,
        callbacks=[es])

# Sauvegarde le réseau sous deux formats : .h5 et SavedModel (dir)
print("Saving...")
model.save("fire_"+SAVE_NAME+"_model.h5")
model.save("fire_"+SAVE_NAME+"_saved_model")

# Plot loss and accuracy
N = len(H.history['loss'])
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("plot_"+SAVE_NAME+".png")

# Confusion Matrix
# Génération de la matrice de confusion non normalisée
# La sortie est un fichier .txt avec les valeurs de la matrice
# Rq : On peut rajouter les instructions de mat2heapmap pour générer directement la matrice
# normée ici.
print('Confusion Matrix')
validation_generator.reset()
Y_preds = model.predict_generator(validation_generator, len(validation_generator))
y_pred = np.argmax(Y_preds, axis=1)
cm=confusion_matrix(validation_generator.classes, y_pred)
np.savetxt("cm_"+SAVE_NAME+".txt",cm,fmt="%s")
