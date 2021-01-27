# PFE 2020 - ML for IoT

Ce projet est une application de détection de feu en utilisant des réseaux de neurones sur une Nvidia Jetson Nano. L'optimisation du réseau se fait grâce à TensorRT.

Version de JetPack : 4.4
TensorFlow : https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html
! pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow

La BDD doit être dans le même répertoire que les fichiers d'entraînement.
Dataset/
| Train/
|   classe1/
|   classeN/
| Test/
|   classe1/
|   classeN/
Results/


Erreur possible : "Cannot allocate memory in static TLS block"
Solution :
! export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

