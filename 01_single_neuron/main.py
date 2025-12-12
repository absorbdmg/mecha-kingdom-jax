import jax.numpy as jnp

def relu(x):
    return jnp.maximum(0, x)

def layer(inputs, weights, biases):
    return jnp.dot(inputs, weights) + biases

def mse_loss(prediction, target):
    return jnp.mean((prediction - target) ** 2)

target = jnp.array([1.0])  # Example target for loss calculation

# Input: 2 values (like XOR inputs)
inputs = jnp.array([1.0, 0.0])

# Layer 1: 2 inputs → 3 neurons
weights1 = jnp.array([
    [0.5, 0.2, -0.3],
    [0.1, -0.4, 0.6],
])
biases1 = jnp.array([0.0, 0.0, 0.0])

# Layer 2: 3 inputs → 1 neuron
weights2 = jnp.array([
    [0.7],
    [-0.2],
    [0.5],
])
biases2 = jnp.array([0.0])

# Forward pass
hidden = layer(inputs, weights1, biases1)
print(f"After layer 1: {hidden}")

hidden = relu(hidden)
print(f"After ReLU:    {hidden}")

output = layer(hidden, weights2, biases2)
print(f"After layer 2: {output}")

error = mse_loss(output, target)
print(f"Loss:          {error}")