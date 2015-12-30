import tensorflow as tf
import numpy as np

from hamiltonian_monte_carlo import hmc

def gaussian_log_posterior(x, covariance):
    """Evaluate the unormalized log posterior from a zero-mean
    Gaussian distribution, with the specifed covariance matrix
    
    Parameters
    ----------
    x : tf.Variable
        Sample ~ target distribution
    covariance : tf.Variable N x N 
        Covariance matrix for N-dim Gaussian

        For diagonal - [[sigma_1^2, 0], [0, sigma_2^2]]

    Returns
    -------
    logp : float
        Unormalized log p(x)
    """
    covariance_inverse = tf.matrix_inverse(covariance)

    xA = tf.matmul(x, covariance_inverse)
    xAx = tf.matmul(xA, tf.transpose(x))
    return xAx / 2.0


def gaussian_log_posterior_diagonal(x, sigma=1):
    covariance = tf.constant(
            np.array([
                [sigma, 0],
                [0, sigma]]), 
            dtype=tf.float32
        )

    return gaussian_log_posterior(x, covariance)

def gaussian_log_posterior_correlated(x, correlation=0.8):
    covariance = tf.constant(
            np.array([
                [1.0, correlation],
                [correlation, 1.0]]), 
            dtype=tf.float32
        )

    return gaussian_log_posterior(x, covariance)


if __name__ == '__main__':

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

        print()
        print("Drawing from a diagonal Gaussian...")
        session.run(tf.initialize_all_variables())
        for i in xrange(num_samples):
            sample = session.run(
                hmc(initial_x, 
                    log_posterior=gaussian_log_posterior_correlated, 
                    step_size=0.1, 
                    num_steps=10)
            )
            print(sample)