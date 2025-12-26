import jax
import jax.numpy as jnp

learning_rate = 0.5
epochs = 1000

def relu(x):
    return jnp.maximum(0, x)

def mse_loss(params, X, y):
    weights1, biases1, weights2, biases2 = params
    hidden = relu(jnp.dot(X, weights1) + biases1)
    predictions = jnp.dot(hidden, weights2) + biases2
    return jnp.mean((predictions - y) ** 2)

def update_params(params, grads, lr):
    return tuple(p - lr * g for p, g in zip(params, grads))

# XOR data
X = jnp.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

y = jnp.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])

# Initialize parameters
weights1 = jnp.array([[0.5, 0.2, -0.3],
                      [0.1, -0.4, 0.6]])
biases1 = jnp.array([0.0, 0.0, 0.0])
weights2 = jnp.array([[0.7], [-0.2], [0.5]])
biases2 = jnp.array([0.0])

params = (weights1, biases1, weights2, biases2)

# Training loop - all examples at once
for epoch in range(epochs):
    loss = mse_loss(params, X, y)
    grads = jax.grad(mse_loss)(params, X, y)
    params = update_params(params, grads, learning_rate)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

# Final predictions
weights1, biases1, weights2, biases2 = params
hidden = relu(jnp.dot(X, weights1) + biases1)
predictions = jnp.dot(hidden, weights2) + biases2

print(f"\nFinal Predictions:")
for i in range(4):
    print(f"  {X[i]} -> {predictions[i][0]:.4f} (target: {y[i][0]})")


print(f"weights1:\n{weights1}")
print(f"weights2:\n{weights2}")