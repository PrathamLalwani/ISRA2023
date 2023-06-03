from params import params
from myPINN import main
from analytic_solution import oscillator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def plot_error(path, filename, errors):
    errors = [[x,y] for x,y,_,_ in errors]
    plt.scatter([np.log2(x) for x, y in errors], [np.log2(y) for x, y in errors], color="blue", marker="x")
    a, b = np.polyfit(np.log2([x for x, y in errors]), np.log2([y for x, y in errors]), 1)
    plt.plot([np.log2(x) for x, y in errors], a * np.log([x for x, y in errors]) + b, color="red")
    plt.title("weight vs Error")
    plt.xlabel("log(collocation points)")
    plt.ylabel("log(Error)")
    plt.legend(["Slope = {}".format(a)])
    plt.savefig(path + filename + "_loglog.jpg")
    plt.close()




(
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
    in_dim,
    out_dim,
    activation,
    optimizer,
    lr,
    random_seed,
    model_filename,
    loss_filename
) = params()
errors = []
widths = [20,30,40,50,60]
# for depth in range(3, 8):
#     for width in widths:
weights = tf.linspace(tf.constant(0.1, dtype=tf.float64), 10, 13)
# for depth in range(3, 8):
#     for width in widths:
for collocation_points in [2**i for i in range(3, 10)]:
    t = tf.reshape(tf.linspace(l, u, 1000),[-1,1])
    # collocation_points = 2**i
    t_ =tf.reshape(tf.linspace(l, u, collocation_points),[-1,1])
    model_filename = f"model_without_hamiltonian_collocation_points_{collocation_points}"
    loss_filename = f"model_without_hamiltonian_loss_collocation_points_{collocation_points}"
    model = main(
        EPOCHS, l, u, k, m, collocation_points, t0, x0, x_0,w_geo, path, width, depth, in_dim, out_dim, activation, optimizer, lr, random_seed,model_filename,loss_filename
    )
    # analytic_sol = tf.reshape(oscillator(x0, x_0, t, k, m), [-1, 1])
    # analytic_sol_ = tf.reshape(oscillator(x0, x_0, t_, k, m), [-1, 1])
    # assert analytic_sol.shape == model(t).shape
    # error_across_t = tf.reduce_mean(tf.square(analytic_sol - model(t)))
    # error_on_collocation_pts = tf.reduce_mean(tf.square(analytic_sol_ - model(t_)))
    # error_at_t_l = tf.reduce_mean(tf.square(analytic_sol[-1] - model(t)[-1]))
    # errors.append([collocation_points,error_at_t_l.numpy() ,error_across_t.numpy(),error_on_collocation_pts.numpy()])
    
# pd.DataFrame(errors,columns = ["collocation_points","error_at_t_1","error_across_t_1000_points", "error_on_collocation_points"]).to_csv(path + "collocation_points_vs_error_with_hamiltonian.csv")    
# pd.DataFrame(errors,columns=["width","depth","error"]).to_csv(path + "collocation_points_vs_error_with_hamiltonian_with_weight.csv")
# plot_error(path, "collocation_points_vs_error_with_hamiltonian.jpg", errors)

