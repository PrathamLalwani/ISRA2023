import tensorflow as tf
import os
from analytic_solution import oscillator


EPOCHS = 20000
l = 0.0
u = 1.
k = 20.0
m = 1.0
collocation_points = 101
t0 = 0.0
x_0 = 0.0
x0 = 1.0
path = os.path.join(os.getcwd(), "model")
width = 10
depth = 10
input_dim = 3
output_dim = 1
activation = 'tanh'
optimizer = "Adam"
learning_rate = 0.001
random_seed = 123
w_geo = 1.
model_filename = "model"
loss_filename = "loss.jpg"  

def params():
    return (
        EPOCHS,
        l,
        u,
        k,
        m,
        collocation_points,
        t0,
        x0,
        x_0,
        w_geo,
        path,
        width,
        depth,
        input_dim,
        output_dim,
        activation,
        optimizer,
        learning_rate,
        random_seed,
        model_filename,
        loss_filename,
    )
