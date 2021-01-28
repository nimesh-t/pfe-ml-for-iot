# PFE de la VAP SEM (Télécom SudParis 2020)  - ML for IoT



Ce projet est une application de détection de feu en utilisant des réseaux de neurones sur une Nvidia Jetson Nano. L'optimisation du réseau se fait grâce à TensorRT.

Python 3.6.9\
Version de JetPack : 4.4\
[Get Started Jetson Nano](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit) \
TensorFlow : https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html
```bash
pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow
```
Le fichier requirements.txt contient les librairies nécessaires ainsi que leurs versions.

# Lancement
La BDD doit être dans le même répertoire que les fichiers d'entraînement.\
Dataset/\
| Train/\
|   classe1/\
|   classeN/\
| Test/\
|   classe1/\
|   classeN/\
Results/\
Inference/
| test_images/\

Les scripts MobileNet.py, ResNet50.py ou Xception.py génère le modèle et les courbes d'entraînement.\
Le script MobileNet_inference.py renvoie les résultats sur une image dans  Inference/test_images/.
Le script MobileNet_inference_trt.py génère un modèle optimisé (FP16) avec TensorRT et renvoie les résultats d'une prédiction avec ce modèle. La deuxième partie ne fonctionne pas encore.
Le script MobileNet_infer_cam.py effectue des prédictions sur des trames issues de la caméra Raspberry Pi Camera V2. Le code utilisé pour la partie caméra vient de : https://github.com/JetsonHacksNano/CSI-Camera .

Erreur possible : "Cannot allocate memory in static TLS block"\
Solution :
```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```

Lien vers la demo : https://youtu.be/DOA3oGs7W9g
