import tensorflow as tf

learning_rate = 0.1


def model(inputs, W, b):
    return tf.matmul(inputs, W) + b


def mean_squared_error(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)


@tf.function(jit_compile=True)
def training_step(inputs, targets, W, b):
    with tf.GradientTape() as tape:
        predictions = model(inputs, W, b)
        loss = mean_squared_error(predictions, targets)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss


def get_initializers(input_dim, output_dim):
    W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
    b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))
    return W, b