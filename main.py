import numpy as np
import matplotlib.pyplot as plt

import linear_classifier_with_jax as linear_classifier


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

    W, b = linear_classifier.get_initializers(input_dim, output_dim)
    state = (W, b)

    for step in range(40):
        loss, W, b = linear_classifier.training_step(inputs, targets, W, b)
        state = (W, b)
        print(f"Loss at step {step + 1}: {loss:.4f}")

    plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
    plt.show()

    predictions = linear_classifier.model(inputs, W, b)
    x = np.linspace(-1, 4, 100)
    y = -W[0] / W[1] * x + (0.5 - b) / W[1]
    plt.plot(x, y, "-r")
    plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
    plt.show()