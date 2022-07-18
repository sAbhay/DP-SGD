import mnist
import cifar

def get_model_fn(model_type: str, model_kwargs):
    if model_type == 'mnist':
        return mnist.get_mnist_model_fn(**model_kwargs)
    elif model_type == 'cifar':
        return cifar.get_cifar_model_fn(**model_kwargs)
    else:
        raise ValueError(f'Unknown model type: {model_type}')