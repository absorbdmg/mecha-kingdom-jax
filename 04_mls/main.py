import jax
import jax.numpy as jnp


def f(x):
    u = jnp.array(
        [
            x[0] ** 6 * x[1] ** 4 * x[2] ** 9 * x[3] ** 2,
            x[0] ** 2 * x[1] ** 3 * x[2] ** 5 * x[3] ** 3,
            x[0] ** 5 * x[1] ** 7 * x[2] ** 7 * x[3] ** 6,
        ]
    )
    return u


evaluation_point = jnp.array([1.0, 0.5, 1.5, 2.0])

# output
print(f(evaluation_point))

# Jacobian df/dx
full_jacobian = jax.jacfwd(f)(evaluation_point)

print(full_jacobian.shape)

# Jacobian-Vector Product df/dx @ v
multiplication_point = jnp.array([0.2, 0.3, 0.4, 0.8])

# naive jvp approach:
print(full_jacobian @ multiplication_point)


# JAX.jvp appraoch:
# Using a JAX function to obtain the result of the jvp without explicitly computing the Jacobian matrix
# jax.jvp(f, primals, tangents)
f_eval, jvp_eval = jax.jvp(f, (evaluation_point,), (multiplication_point,))

print(f_eval)
print(jvp_eval)
