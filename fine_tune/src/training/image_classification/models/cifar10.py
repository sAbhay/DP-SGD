from wideresnet import WideResNet
from ..data import cifar10

def get_model(depth=16, width=4, dropout_rate=0.0):
    return WideResNet(depth=depth, num_classes=len(cifar10.CLASSES), widen_factor=width, dropRate=dropout_rate)