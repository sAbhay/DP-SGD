import jax
import jax.numpy as jnp

activations_dict = {
    # Regular activations.
    'identity': lambda x: x,
    'celu': jax.nn.celu,
    'elu': jax.nn.elu,
    'gelu': jax.nn.gelu,
    'glu': jax.nn.glu,
    'leaky_relu': jax.nn.leaky_relu,
    'log_sigmoid': jax.nn.log_sigmoid,
    'log_softmax': jax.nn.log_softmax,
    'relu': jax.nn.relu,
    'relu6': jax.nn.relu6,
    'selu': jax.nn.selu,
    'sigmoid': jax.nn.sigmoid,
    'silu': jax.nn.silu,
    'swish': jax.nn.silu,
    'soft_sign': jax.nn.soft_sign,
    'softplus': jax.nn.softplus,
    'tanh': jnp.tanh,

    # Scaled activations.
    'scaled_celu': lambda x: jax.nn.celu(x) * 1.270926833152771,
    'scaled_elu': lambda x: jax.nn.elu(x) * 1.2716004848480225,
    'scaled_gelu': lambda x: jax.nn.gelu(x) * 1.7015043497085571,
    'scaled_glu': lambda x: jax.nn.glu(x) * 1.8484294414520264,
    'scaled_leaky_relu': lambda x: jax.nn.leaky_relu(x) * 1.70590341091156,
    'scaled_log_sigmoid': lambda x: jax.nn.log_sigmoid(x) * 1.9193484783172607,
    'scaled_log_softmax': lambda x: jax.nn.log_softmax(x) * 1.0002083778381348,
    'scaled_relu': lambda x: jax.nn.relu(x) * 1.7139588594436646,
    'scaled_relu6': lambda x: jax.nn.relu6(x) * 1.7131484746932983,
    'scaled_selu': lambda x: jax.nn.selu(x) * 1.0008515119552612,
    'scaled_sigmoid': lambda x: jax.nn.sigmoid(x) * 4.803835391998291,
    'scaled_silu': lambda x: jax.nn.silu(x) * 1.7881293296813965,
    'scaled_swish': lambda x: jax.nn.silu(x) * 1.7881293296813965,
    'scaled_soft_sign': lambda x: jax.nn.soft_sign(x) * 2.338853120803833,
    'scaled_softplus': lambda x: jax.nn.softplus(x) * 1.9203323125839233,
    'scaled_tanh': lambda x: jnp.tanh(x) * 1.5939117670059204,
}