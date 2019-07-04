#Computer Vision Models

- Architecure

  - **(alexnet)** [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
  - **(googlenet)** [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
  - **(vgg)** [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
  - **(resnet)** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
  - **(sppnet)** [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729)
  - **(mobilenet_v1)** [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
  

## Results on CIFAR

| architecture          | params | batch size | epoch | C10 test acc (%) | C100 test acc (%) |
| :-------------------- | :----: | :--------: | :---: | :--------------: | :---------------: |
| alexnet               | 24.7M  |    128     |  250  |      82.70       |         -         |
| inception_v1          |  7.2M  |    128     |  250  |      91.52       |         -         |
| vgg19_bn              | 38.9M  |    128     |  250  |      93.24       |         -         |
| mobilenet_v1          |  3.2M  |    128     |  200  |      86.01       |         -         |
