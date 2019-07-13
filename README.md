# Computer Vision Models

- Architecure

  - **(alexnet)** [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
  - **(googlenet)** [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
  - **(vgg)** [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
  - **(resnet)** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
  - **(sppnet)** [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729)
  - **(mobilenet_v1)** [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
  - **(psmnet)** [Pyramid Stereo Matching Network](https://arxiv.org/abs/1803.08669)


## Requirements and Usage 

### Requirements

- Python >= 3.5
- PyTorch >= 1.0.1.post2
- torchvision >= 0.2.2.post3
- TensorFlow/Tensorboard (if you want to use the tensorboard for **visualization**)
- Other dependencies (pyyaml, easydict, tensorboardX)

```bash
pip install -r requirements.txt
```

### Usage 

simply run the cmd for the training:

```bash
## 1 GPU for psm_net
CUDA_VISIBLE_DEVICES=0 python -u train_psmnet.py --work-path ./experiments/psmnet/kitti

## resume from ckpt
CUDA_VISIBLE_DEVICES=0 python -u train_psmnet.py --work-path ./experiments/psmnet/kitti --resume

## 1 GPU for vgg
CUDA_VISIBLE_DEVICES=0 python -u train_simple.py --work-path ./experiments/vgg/cifar10

## resume from ckpt
CUDA_VISIBLE_DEVICES=0 python -u train_simple.py --work-path ./experiments/vgg/cifar10 --resume

## 2 GPUs for inception_v1
CUDA_VISIBLE_DEVICES=0,1 python -u train_simple.py --work-path ./experiments/inception_v1/cifar10

## 4 GPUs for resnet
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train_simple.py --work-path ./experiments/resnet/cifar10
``` 

We use yaml file ``config.yaml`` to save the parameters, check any files in `./experimets` for more details.  
You can see the training curve via tensorboard, ``tensorboard --logdir path-to-event --port your-port``. (Not verified)  
The training log will be dumped via logging, check ``log.txt`` in your work path.

## Results on CIFAR

| architecture          | params | batch size | epoch | C10 test acc (%) | C100 test acc (%) |
| :-------------------- | :----: | :--------: | :---: | :--------------: | :---------------: |
| alexnet               | 24.7M  |    128     |  250  |      82.70       |         -         |
| inception_v1          |  7.2M  |    128     |  250  |      91.52       |         -         |
| vgg19_bn              | 38.9M  |    128     |  250  |      93.24       |         -         |
| resnet                |  x.xM  |    128     |  250  |      xx.xx       |         -         |
| mobilenet_v1          |  3.2M  |    128     |  200  |      86.01       |         -         |
