# PFE 2020 - ML for IoT

## [TensorRT](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#usingtftrt)
* Quantisation
* Pruning
* Clustering

## Training example
`jetson-inference/python/training/classification/train.py`

## Transfer learning
### Manually populate dataset using camera-capture from jetson-inference
```bash
mkdir datasets
cd ~/datasets
mkdir <dataset_name>
cd <dataset_name>
touch labels.txt
echo “label1” >> labels.txt
echo “label2” >> labels.txt

camera-capture --camera=0 --width=640 --height=480
```

## Monitoring Jetson Nano
`tegrastats` - shows CPU, memory, GPU usage\
`jtop` - GUI stats (`pip install jetson-stats`)

## Useful links
[Getting Started with AI on Jetson Nano](https://courses.nvidia.com/courses/course-v1:DLI+S-RX-02+V2/about) \
[Camera usage documentation](https://developer.download.nvidia.com/embedded/L4T/r24_Release_v2.0/Docs/L4T_Tegra_X1_Multimedia_User_Guide_Release_24.2.pdf?ORzXaY-aQWa-QsQCPbqN8XcwbHMxXI_oyRtg_2hkGETt-YUUTyD_YFx5YJpeOhkRp5oHxhHc88Q4GmstgGw3na8H_xqlm1CCvTIr6zLKpQyxQXL0yN26KTMH8xOMx6pdeCjUSo5Vja2okulw2mSJPtduOxs-tWHqxUxtM32Lf1do5HPmKzqHhTsRdmmnUSkm9ynPSv4)\
[TensorRT with PyTorch](https://www.learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/)\
[Package download links](https://elinux.org/Jetson_Zoo)\
[int8 Quantisation](https://www.mathworks.com/company/newsletters/articles/what-is-int8-quantization-and-why-is-it-popular-for-deep-neural-networks.html)\
[Federated Learning](https://medium.com/@ODSC/what-is-federated-learning-99c7fc9bc4f5)
