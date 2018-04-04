# ResNet in PyTorch
ResNet training script in PyTorch

Reference:  
[1] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## Requirements

First install [PyTorch](https://pytorch.org), then install other Python packages:

```
pip install -r requirements.txt
```

To train diracnet, configure parameters in train.sh and run:

```bash
sh train.sh
```

## Result
On ILSVRC2012, the Top-1/Top-5 (%) for ResNet-18 is 70.0/ .

## Credit
This implementation is initially inspired by:
- [bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification)
- [pytorch/vision/torchvision/models/resnet.py](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)

