import numpy as np
import tensorflow as tf
def oscillator(x0, x_0, t, k, m):
    omega = np.sqrt(k / m)
    x_max = tf.sqrt(x0**2 + (x_0/omega)**2)
    phase_diff_x = tf.acos(x0/x_max)
    return x_max * np.cos(np.sqrt(k / m) * t + phase_diff_x)
