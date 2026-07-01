import numpy as np
import jax
from jax import numpy as jnp

learning_rate = 0.1

def mean_squared_error(targets, predictions):
    per_sample_losses = jnp.square(targets - predictions)
    return jnp.mean(per_sample_losses)

def model(inputs, W, b):
    return jnp.matmul(inputs, W) + b

def compute_loss(state, inputs, targets):
    W, b = state
    predictions = model(inputs, W, b)
    loss = mean_squared_error(targets, predictions)
    return loss

grad_fn = jax.value_and_grad(compute_loss)

@jax.jit
def training_step( inputs, targets, W, b):
    loss, grads = grad_fn((W, b), inputs, targets)
    grad_wrt_W, grad_wrt_b = grads
    W = W - grad_wrt_W * learning_rate
    b = b - grad_wrt_b * learning_rate
    return loss, W, b

def get_initializers(input_dim, output_dim):
    W = jax.numpy.array(np.random.uniform(size=(input_dim, output_dim)))
    b = jax.numpy.array(np.zeros(shape=(output_dim,)))
    return W, b