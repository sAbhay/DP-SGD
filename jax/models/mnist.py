from .layers import WSConv2D
import haiku as hk
import jax

class CNN(hk.Module):
    def __init__(self, overparameterised=True, groups=8, weight_standardisation=True, depth=2, output_classes=10):
        super(CNN, self).__init__()
        self.multiplier = 2 if overparameterised else 2
        self.groups = groups
        if weight_standardisation:
            self.conv_fn = WSConv2D
        else:
            self.conv_fn = hk.Conv2D
        self.depth = depth
        assert (depth % 2 == 0) and (depth > 3), "Depth must be even and at least 4"
        self.groups = groups
        self.output_classes = output_classes

    def __call__(self, features, **_):
        net = features
        for i in range(self.depth // 2 - 1):
            net = self.conv_fn(16*self.multiplier, (8, 8), padding='SAME', stride=(2, 2), name='conv_%d' % i)(net)
            net = jax.nn.relu(net),
            net = hk.MaxPool(2, 1, padding='VALID')(net)
        if self.groups > 0:
            net = hk.GroupNorm(self.groups)(net)
            net = jax.nn.relu(net),
            net = hk.MaxPool(2, 1, padding='VALID')(net)
        for i in range(self.depth // 2 - 1):
            net = self.conv_fn(32*self.multiplier, (4, 4), padding='SAME', stride=(2, 2), name='conv_%d' % (i+self.depth//2))(net)
        if self.groups > 0:
            net = hk.GroupNorm(self.groups)(net)
        net = hk.Flatten()(net)
        net = hk.Linear(32)(net)
        net = jax.nn.relu(net)
        net = hk.Linear(self.output_classes)(net)
        return net


def get_mnist_model_fn(overparameterised=True, groups=8, weight_standardisation=True, depth=4, output_classes=10):
    multiplier = 2 if overparameterised else 1
    if weight_standardisation:
        conv_fn = WSConv2D
    else:
        conv_fn = hk.Conv2D
    def mnist_model_fn(features, **_):
        model = CNN(overparameterised, groups, weight_standardisation, depth, output_classes)
        return model(features)
    def mnist_model_fn_seq(features, **_):
        layers = []
        for i in range(depth // 2 - 1):
            layers.append(conv_fn(16 * multiplier * (2 ** i), (8, 8), padding='SAME', stride=(2, 2)))
            layers.append(jax.nn.relu)
            if groups > 0:
                layers.append(hk.GroupNorm(groups))
            layers.append(hk.MaxPool(2, 1, padding='VALID'))  # matches stax
            layers.append(conv_fn(32 * multiplier * (2 ** i), (4, 4), padding='SAME', stride=(2, 2)))
            layers.append(jax.nn.relu)
            if groups > 0:
                layers.append(hk.GroupNorm(groups))
            layers.append(hk.MaxPool(2, 1, padding='VALID'))  # matches stax
        layers.append(hk.Flatten())
        layers.append(hk.Linear(32))
        layers.append(jax.nn.relu)
        layers.append(hk.Linear(output_classes))
        model = hk.Sequential(layers)
        return model(features)
    return mnist_model_fn_seq
