import jax
import jax.numpy as jnp


def relu(x):
    return jnp.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def forward(params, X):
    weights1, biases1, weights2, biases2 = params
    hidden = relu(jnp.dot(X, weights1) + biases1)
    output = sigmoid(jnp.dot(hidden, weights2) + biases2)
    return output


# XOR data
X = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = jnp.array([[0.0], [1.0], [1.0], [0.0]])

# Initialize params (same as before)
weights1 = jnp.array([[0.5, 0.2, -0.3], [0.1, -0.4, 0.6]])
biases1 = jnp.array([0.0, 0.0, 0.0])
weights2 = jnp.array([[0.7], [-0.2], [0.5]])
biases2 = jnp.array([0.0])


params = (weights1, biases1, weights2, biases2)

# Test forward pass
predictions = forward(params, X)
print("Predictions (before training):")
print(predictions)

