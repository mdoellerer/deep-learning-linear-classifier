import numpy as np
import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt


def model(inputs, W, b):
    return jnp.matmul(inputs, W) + b


def mean_squared_error(targets, predictions):
    per_sample_losses = jnp.square(targets - predictions)
    return jnp.mean(per_sample_losses)


def compute_loss(state, inputs, targets):
    W, b = state
    predictions = model(inputs, W, b)
    loss = mean_squared_error(targets, predictions)
    return loss

learning_rate = 0.1
grad_fn = jax.value_and_grad(compute_loss)


@jax.jit
def training_step(inputs, targets, W, b):
    loss, grads = grad_fn((W, b), inputs, targets)
    grad_wrt_W, grad_wrt_b = grads
    W = W - grad_wrt_W * learning_rate
    b = b - grad_wrt_b * learning_rate
    return loss, W, b


if __name__ == '__main__':
    num_samples_per_class = 1000
    negative_samples = np.random.multivariate_normal(
        mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
    )
    positive_samples = np.random.multivariate_normal(
        mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
    )

    inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

    targets = np.vstack(
        (
            np.zeros((num_samples_per_class, 1), dtype="float32"),
            np.ones((num_samples_per_class, 1), dtype="float32"),
        )
    )
    input_dim = 2
    output_dim = 1

    W = jax.numpy.array(np.random.uniform(size=(input_dim, output_dim)))
    b = jax.numpy.array(np.zeros(shape=(output_dim,)))
    state = (W, b)
    for step in range(40):
        loss, W, b = training_step(inputs, targets, W, b)
        print(f"Loss at step {step + 1}: {loss:.4f}")

    plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
    plt.show()

    predictions = model(inputs, W, b)
    x = np.linspace(-1, 4, 100)
    y = -W[0] / W[1] * x + (0.5 - b) / W[1]
    plt.plot(x, y, "-r")
    plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
    plt.show()