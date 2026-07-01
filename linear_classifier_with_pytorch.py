import torch

learning_rate = 0.1

def model(inputs, W, b):
    inputs = torch.from_numpy(inputs)
    return torch.matmul(inputs, W) + b


def mean_squared_error(targets, predictions):
    targets = torch.from_numpy(targets)
    per_sample_losses = torch.square(targets - predictions)
    return torch.mean(per_sample_losses)


def training_step(inputs, targets, W, b):
    predictions = model(inputs, W, b)
    loss = mean_squared_error(targets, predictions)
    loss.backward()
    grad_loss_wrt_W, grad_loss_wrt_b = W.grad, b.grad
    with torch.no_grad():
        W -= grad_loss_wrt_W * learning_rate
        b -= grad_loss_wrt_b * learning_rate
    W.grad = None
    b.grad = None
    return loss


def get_initializers(input_dim, output_dim):
    W = torch.rand(input_dim, output_dim, requires_grad=True)
    b = torch.zeros(output_dim, requires_grad=True)
    return W, b