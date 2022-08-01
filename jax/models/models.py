from .mnist import get_mnist_model_fn
from .cifar import get_cifar_model_fn
from .wideresnet import get_widresnet_fn

def get_model_fn(model_type: str, model_kwargs):
    if model_type == 'cnn':
        return get_mnist_model_fn(**model_kwargs)
    elif model_type == 'wideresnet':
        return get_cifar_model_fn(**model_kwargs)
    elif model_type == 'wideresnet2':
        return get_widresnet_fn(**model_kwargs)
    else:
        raise ValueError(f'Unknown model type: {model_type}')