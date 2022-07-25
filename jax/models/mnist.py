from .layers import WSConv2D
import haiku as hk
import jax

class CNN():
    def __init__(self, overparameterised=True, groups=8, weight_standardisation=True, depth=2, output_classes=10):
        self.multiplier = 2 if overparameterised else 2
        self.groups = groups
        if weight_standardisation:
            self.conv_fn = WSConv2D
        else:
            self.conv_fn = hk.Conv2D
        self.depth = depth
        assert ((depth-1) % 2 == 0) and (depth > 2), "Depth must be odd and greater than 2"
        self.groups = groups
        self.output_classes = output_classes

    def __call__(self, features, **_):
        net = features
        for i in range(self.depth // 2):
            net = self.conv_fn(16*self.multiplier, (8, 8), padding='SAME', stride=(2, 2), name='conv_%d' % i)(net)
        if self.groups > 0:
            net = hk.GroupNorm(self.groups)(net)
        for i in range(self.depth // 2):
            net = self.conv_fn(32*self.multiplier, (4, 4), padding='SAME', stride=(2, 2), name='conv_%d' % (i+self.depth//2))(net)
        if self.groups > 0:
            net = hk.GroupNorm(self.groups)(net)
        net = hk.Flatten(net)
        net = hk.Linear(32)(net)
        net = jax.nn.relu(net)
        net = hk.Linear(self.output_classes)(net)
        return net


def get_mnist_model_fn(overparameterised=True, groups=8, weight_standardisation=True, depth=2, output_classes=10):
    def mnist_model_fn(features, **_):
        model = CNN(overparameterised, groups, weight_standardisation, depth, output_classes)
        return model(features)
    return mnist_model_fn
