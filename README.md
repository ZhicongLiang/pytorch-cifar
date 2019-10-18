# Introduction
This is for Homework in MATH6450 HKUST.

The structure is borrowed from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) and the DenseNet is borrowed from [densenet-pytorch](https://github.com/andreasveit/densenet-pytorch).

We achieve testing accuracy of 94.16% and 95.43% with DenseNet-40-12 and DensetNet-85-12. And we achieve a testing accuracy of 97.36% with EfficientNet-B0 initialized with ImageNet pretrained weight.

# Result:

![accuracy](https://github.com/ZhicongLiang/pytorch-cifar/blob/master/accuracy.png)

![training curve of model trained from scratch](https://github.com/ZhicongLiang/pytorch-cifar/blob/master/training_curves.png)

![training curve of model trained from pretrianed weight](https://github.com/ZhicongLiang/pytorch-cifar/blob/master/training_curves_2.png)

# Usage

### to train EfficientNet-b0 from scratch:

python main.py --epochs 400 --gpu 0 --wd 4e-5 --lr 0.1 --bs 64 --model efficientnet-b0

### to train EfficientNet-b0 from pretrained weight:

python main.py --epochs 10 --gpu 0 --wd 4e-5 --lr 0.01 --bs 64 --model efficientnet-b0 --pretrained

### to train DenseNet-40-12 from scratch:

python main.py --model DenseNet3 --gpu 0 --layers 40

### to train DenseNet-85-12 from scratch:

python main.py --model DenseNet3 --gpu 0 --layers 85


### to train MobileNetV2 from scratch:

python main.py --model MobileNetV2 --epochs 400 --lr 0.1 --bs 128 --wd 4e-5 --gpu 0

# All the weights of model mentioned above can be download from:

[here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliangak_connect_ust_hk/EoAqUUtVc3lIuE5p7zOy7oQBylimjiy3OGAOLj1Z_TJhMQ?e=QQVfXY)



