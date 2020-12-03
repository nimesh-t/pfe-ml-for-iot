
## Générer .csv contenant le nom des fichiers + leurs dimensions + bounding box (opt)
```
Dataset
| Train
| | Class_1
| | Class_x
| Test
| | Class_1
| | Class_x
```


```python
! sudo apt-get install libimage-exiftool-perl
! exiftool -csv -ext jpg -ImageWidth -ImageHeight /MyFolder > output.csv
```

## Générer TFRecords à partir des fichiers .csv pour la BDD Train et Test


```python
import csv
import tensorflow as tf
import dataset_util

IMAGE_EXT = ".jpg"
IMAGE_FORMAT = b'jpg'

def class_to_id(class_name):
    if class_name == 'Fire':
        return 1
    if class_name == 'Smoke':
        return 2
    if class_name == 'Neutral':
        return 3
    else:
        none
        
        
def create_record_from_csv(input_csv,output_record):
    
    writer = tf.python_io.TFRecordWriter(output_record)
    
    with open(input_csv) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_name = (row['SourceFile']).split('/')[3]
            filename = (row['SourceFile'])
            # Si on veut uniquement le nom du fichier (pas le chemin) => modifier argument de GFile
            # filename = (row['SourceFile']).split('/')[4]
            width = row['ImageWidth']
            height = row['ImageHeight']
            
            # append ration  (eg. xmin/width)
            # ici, on suppose qu'il n'y a pas de bounding box (== toute l'image)
            xmins.append(1)
            xmaxs.append(1)
            ymins.append(1)
            ymaxs.append(1)
            classes_name.append(class_name.encode('utf8'))
            classes.append(class_to_id(class_name))
        
        with tf.gfile.GFile(filename, 'rb') as fid:
                encoded_jpg = fid.read()
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature(IMAGE_FORMAT),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_name),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
            }))
            
            writer.write(tf_example.SerializeToString())
    writer.close()             
    output_path = os.path.join(os.getcwd(), path_output)
    print('Successfully created the TFRecords: {}'.format(output_path))
    
create_record_from_csv('test_images.csv','test_images.record')
create_record_from_csv('test_images.csv','train_images.record')
```

# Créer le label map


```python
item {
    id : 1
    name: 'Fire'
}
item {
    id : 2
    name: 'Smoke'
}
item {
    id : 1
    name: 'Neutral'
}
```

## TF Models


```python
! git clone https://github.com/tensorflow/models.git
    
! cd models/research
# Compile protos.
! protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
! cp object_detection/packages/tf2/setup.py .
! python -m pip install .
```

## Configurer les paramètres d'entraînement dans le .config correspondant au réseau utilisé du Model Zoo de TF
[TF1_Model_Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)

[TF2_Model_Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

### Paramètres à vérifier/modifier
- num_classes
- input_path
- label_map_path
- max_total_detections (à 1 dans notre cas)
- fine_tune_checkpoint


```python
! python train.py --logtostderr --train_dir=train/ --pipeline_config_path=XXXX.config

# model_main_tf2.py recommandé au lieu de train.py
```


```python
! tensorboard --logdir=training/train
```

## Générer le modèle d'inférence


```python
! python exporter_main_v2.py --trained_checkpoint_dir=train --pipeline_config_path=train/XXXX.config --output_directory inference_graph
```

## Optimisation avec TensorRT

### Génération du frozen graph


```python
from tf_trt_models.detection import build_detection_graph

frozen_graph, input_names, output_names = build_detection_graph(
    config=config_path,
    checkpoint=checkpoint_path
)
```

### Modèle optimisé


```python
import tensorflow.contrib.tensorrt as trt

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)
```

## Inférence 

[Inférence avec une image](https://github.com/NVIDIA-AI-IOT/tf_trt_models/blob/master/examples/classification/classification.ipynb)

[Inférence avec la caméra RPi](https://github.com/memillopeloz/JetsonNano-RPICam/blob/master/Object_detection_picamera.py)
