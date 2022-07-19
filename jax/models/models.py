from .mnist import get_mnist_model_fn
from .cifar import get_cifar_model_fn

def get_model_fn(model_type: str, model_kwargs):
    if model_type == 'mnist':
        return get_mnist_model_fn(**model_kwargs)
    elif model_type == 'cifar10':
        return get_cifar_model_fn(**model_kwargs)
    else:
        raise ValueError(f'Unknown model type: {model_type}')