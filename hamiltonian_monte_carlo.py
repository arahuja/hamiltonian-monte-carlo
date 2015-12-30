import tensorflow as tf
import numpy as np

def kinetic_energy(velocity):
    """Kinetic energy of the current velocity (assuming a standard Gaussian)
        (x dot x) / 2

    Parameters
    ----------
    velocity : tf.Variable
        Vector of current velocity

    Returns
    -------
    kinetic_energy : float
    """
    return 0.5 * tf.square(velocity)

def hamiltonian(position, velocity, energy_function):
    """Computes the Hamiltonian of the current position, velocity pair

    H = U(x) + K(v)

    U is the potential energy and is = -log_posterior(x)

    Parameters
    ----------
    position : tf.Variable
        Position or state vector x (sample from the target distribution)
    velocity : tf.Variable
        Auxiliary velocity variable
    energy_function
        Function from state to position to 'energy'
         = -log_posterior

    Returns
    -------
    hamitonian : float
    """
    return energy_function(position) + kinetic_energy(velocity)

def leapfrog_step(x0, 
                  v0, 
                  log_posterior,
                  step_size,
                  num_steps):

    # Start by updating the velocity a half-step
    v = v0 - 0.5 * step_size * tf.gradients(log_posterior(x0), x0)[0]

    # Initalize x to be the first step
    x = x0 + step_size * v

    for i in xrange(num_steps):
        # Compute gradient of the log-posterior with respect to x
        gradient = tf.gradients(log_posterior(x), x)[0]

        # Update velocity
        v = v - step_size * gradient

        # Update x
        x = x + step_size * v

    # Do a final update of the velocity for a half step
    v = v - 0.5 * step_size * tf.gradients(log_posterior(x), x)[0]

    # return new proposal state
    return x, v

def hmc(initial_x,
        step_size, 
        num_steps, 
        log_posterior):
    """Summary

    Parameters
    ----------
    initial_x : tf.Variable
        Initial sample x ~ p
    step_size : float
        Step-size in Hamiltonian simulation
    num_steps : int
        Number of steps to take in Hamiltonian simulation
    log_posterior : str
        Log posterior (unnormalized) for the target distribution

    Returns
    -------
    sample : 
        Sample ~ target distribution
    """

    v0 = tf.random_normal(initial_x.get_shape())
    x, v = leapfrog_step(initial_x,
                      v0, 
                      step_size=step_size, 
                      num_steps=num_steps, 
                      log_posterior=log_posterior)

    orig = hamiltonian(initial_x, v0, log_posterior)
    current = hamiltonian(x, v, log_posterior)
    p_accept = min(1.0, tf.exp(orig - current))

    if p_accept > np.random.uniform():
        return x
    else:
        return initial_x