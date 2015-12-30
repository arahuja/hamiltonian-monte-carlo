# hamiltonian-monte-carlo

Implementation of Hamiltonian Monte Carlo using Google's TensorFlow

### Examples

In `gaussian_sampler_example.py` there is an example of sampling from either a diagonal, 0-mean Gaussian distribution or a correlated Gaussian distribution.

### Define a log-posterior
To use HMC, you need to define a log-posterior function (unnormalized)

Example: mean=0, correlation=0.8 Gaussian

```python
def gaussian_log_posterior_correlated(x)
    covariance_inverse = tf.matrix_inverse(covariance)

    covariance = tf.constant(
            np.array([
                [1.0, correlation],
                [correlation, 1.0]]), 
            dtype=tf.float32
        )

    Ax = tf.matmul(x, covariance_inverse)
    xAx = tf.matmul(Ax, tf.transpose(x))
    return xAx / 2.0
```

### Drawing samples
    
```python
from hamiltonian_monte_carlo import hmc
    
num_samples = 20

# Number of dimensions for samples
ndim = 2

session = tf.Session()
with session.as_default():
    print("Drawing from a correlated Gaussian...")
    initial_x = tf.Variable(
        tf.random_normal(
            (1, ndim), 
            dtype=tf.float32)
        )
    session.run(tf.initialize_all_variables())
    for i in xrange(num_samples):
        sample = session.run(
            hmc(initial_x, 
                log_posterior=gaussian_log_posterior_correlated, 
                step_size=0.1, 
                num_steps=10)
        )

        print(sample)
```
