import numpy as np
import matplotlib.pyplot as plt

import importlib


if __name__ == '__main__':
    choice = None
    linear_classifier = None
    valid_choice = False

    while not valid_choice:
        print("Choose an option for your Linear Classifier:")
        print("1. Option JAX (linear_classifier_with_jax)")
        print("2. Option Tensor Flow (linear_classifier_with_tensorflow)")
        print("3. Option PyTorch (linear_classifier_with_torch)")

        choice = input("Enter your choice (1, 2, or 3): ").strip()

        module_map = {
            "1": "linear_classifier_with_jax",
            "2": "linear_classifier_with_tensorflow",
            "3": "linear_classifier_with_pytorch",
        }

        if choice in module_map:
            module_name = module_map[choice]
            linear_classifier = importlib.import_module(module_name)
            print(f"Loaded {module_name}")
            valid_choice = True
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


    assert(linear_classifier is not None)

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
        if choice == "1":
            loss, W, b = linear_classifier.training_step(inputs, targets, W, b)
            state = (W, b)
        else:
            loss = linear_classifier.training_step(inputs, targets, W, b)

        print(f"Loss at step {step + 1}: {loss:.4f}")

    plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
    plt.show()

    predictions = linear_classifier.model(inputs, W, b)
    x = np.linspace(-1, 4, 100)

    if choice == "3":
        y = (-W[0] / W[1]).detach().numpy() * x + ((0.5 - b) / W[1]).detach().numpy()
    else:
        y = -W[0] / W[1] * x + (0.5 - b) / W[1]

    plt.plot(x, y, "-r")
    plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
    plt.show()