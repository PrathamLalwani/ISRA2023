import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from analytic_solution import oscillator
from params import params
import sys


class pinn(tf.keras.Model):
    def __init__(
        self,
        width: int,
        depth: int,
        in_dim: int,
        out_dim: int,
        activation: str,
        optimizer: str,
        lr: float,
        x0,
        x_0,
        t0,
        c,
        k,
        m,
        u,
        l,
        w_geo,
        random_seed: int = 12,
    ):
        super(pinn, self).__init__()
        self.width = width
        self.depth = depth
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.x0 = x0
        self.activation = activation
        self.opt = self.get_optimizer(optimizer, lr)
        self.lr = lr
        self.t0 = t0
        self.x0 = x0
        self.u = u
        self.l = l
        self.c = tf.constant(c, dtype = tf.float64)
        self.x_0 = x_0
        self.w_geo = tf.convert_to_tensor(w_geo, dtype=tf.float64)
        self.k = tf.convert_to_tensor(k, dtype=tf.float64)
        self.m = tf.convert_to_tensor(m, dtype=tf.float64)
        tf.random.set_seed(random_seed)
        dense_layer_1 = tf.keras.layers.Dense(self.width, activation=self.activation, dtype=tf.float64)
        dense_layer_2 = tf.keras.layers.Dense(self.width, activation=self.activation, dtype=tf.float64)
        out_dense_layer_1 = tf.keras.layers.Dense(self.out_dim, dtype=tf.float64) 
        out_dense_layer_2 = tf.keras.layers.Dense(self.out_dim, dtype=tf.float64) 
        
        add_layer_1 = tf.keras.layers.Add(dtype=tf.float64)
        add_layer_2 = tf.keras.layers.Add(dtype=tf.float64)
        
        x= tf.keras.Input(shape=(self.in_dim,), dtype=tf.float64)
        pt = tf.keras.layers.Lambda(lambda x: tf.concat(tf.split(x,num_or_size_splits=3,axis=1)[0:2],axis=1),dtype=tf.float64)(x)
        qt = tf.keras.layers.Lambda(lambda x: tf.concat([tf.split(x,num_or_size_splits=3,axis=1)[0],tf.split(x,num_or_size_splits=3,axis=1)[2]],axis=1))(x)
        H_q = dense_layer_2(qt)
        H_q = out_dense_layer_2(H_q)
        H_q = tf.keras.layers.Lambda(lambda x: -x,dtype=tf.float64)(H_q)
        p = tf.keras.layers.Lambda(lambda x: tf.split(x,num_or_size_splits=2,axis=1)[1])(pt)
        q = tf.keras.layers.Lambda(lambda x: tf.split(x,num_or_size_splits=2,axis=1)[1])(qt)
        pp_t = add_layer_1([p, H_q])
        H_p = dense_layer_1(pp_t)
        H_p = out_dense_layer_1(H_p)
        qq_t = add_layer_2([q, H_p])
        output = tf.keras.layers.concatenate([pp_t, qq_t], axis=1,dtype=tf.float64)
        self.model = tf.keras.Model(inputs=x, outputs=output)
        
        
        tf.keras.utils.plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    
    def get_optimizer(self, optimizer, lr):
        if optimizer == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0, nesterov=False)
        elif optimizer == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=0.9, momentum=0.0, centered=False)
        elif optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
        elif optimizer == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(learning_rate=lr, beta_1=0.9, beta_2=0.999)
        elif optimizer == "Nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
        else:
            raise Exception(">>>>> Exception: optimizer not specified correctly")

        return optimizer

    # @tf.function
    # def ODE(self, t):
    #     with tf.GradientTape(persistent=True) as tp:
    #         tp.watch(t)
    #         x = self.predict(t)
    #         x_ = tp.gradient(x, t)
    #     x__ = tp.gradient(x_, t)
    #     del tp
    #     return x, x_, x__ + (self.k / self.m) * x, (self.m * (x_ ** 2)) / (2) + (1 / 2) * self.k * (x**2)
    @tf.function
    def get_loss(self, t,p,q):
        with tf.GradientTape(persistent=True) as tp:
            tp.watch(t)
            tp.watch(p)
            tp.watch(q) 
            xx = self.predict(t,p,q) 
            pp,qq = tf.split(xx,num_or_size_splits=2,axis=1)
            H = ( tf.square(pp))/(2*self.m) + (self.k * tf.square(qq))/2
        #     qq_t = tp.gradient(qq, t)
        # qq_tt = tp.gradient(qq_t, t)
        # phy = tf.square(qq_tt+(self.k/self.m)*qq)
        err_1 = tf.square((pp - p)[1:]/self.c + (self.k*qq[:-1]))
        err_2 = tf.square((qq- q)[1:]/self.c - pp[1:]/self.m)
        H0 = (self.m * self.x_0**2)/2 + (self.k * self.x0**2)/2
        H = (H-H0[0])/H0[0]
        del tp
        # result= self.predict(t,x)
        return  tf.reduce_mean(err_1) + tf.reduce_mean(err_2) + tf.reduce_mean(tf.square(H)) #+ tf.reduce_mean(phy)
    @tf.function
    def call(self,input):
        return self.model(input)

    @tf.function
    def predict(self, t,p,q):
        return self(tf.concat([t,p,q],axis=1))

def train_step(model:pinn, optimizer, t, p,q):
    with tf.GradientTape(persistent=True) as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        loss = model.get_loss(t, p,q)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def plot_loss(path, filename, model, losses, epochs):
    plt.plot(range(epochs), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epochs min_loss = {np.min(losses):.4f} with width={model.width} and depth={model.depth}")
    plt.legend(["Loss"])
    plt.savefig(path + filename + ".jpg")
    plt.close()


def plot_energy_error(path, filename, losses, t, collocation_points, epochs):
    plt.plot(range(epochs),losses )
    plt.xlabel("EPOCHS")
    plt.ylabel("Absolute Energy Error")
    plt.title(f"Energy Error at t = {t.numpy()} vs epochs with collocation points= {collocation_points}")
    plt.legend(["PINN Energy Error"])
    plt.savefig(path + filename + ".jpg")
    plt.close()

def calc_energy_error(model,t,x0,x_0,k,m):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        x = model.predict(t)
    x_ = tape.gradient(x, t)
    H = (m * x_**2) / 2 + (k * x**2) / 2
    H0 = (m * x_0**2) / 2 + (k * x0**2) / 2
    return tf.abs(H - H0)[0]


def main(
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
    loss_filename,
):
    with tf.device("gpu:0"):
        l = tf.constant(l, dtype=tf.float64)
        u = tf.constant(u, dtype=tf.float64)
        k = tf.constant(k, dtype=tf.float64)
        m = tf.constant(m, dtype=tf.float64)
        c = tf.constant((u-l)/(collocation_points-1),dtype=tf.float64)
        t = tf.linspace(l,u,collocation_points)
        t = tf.convert_to_tensor(tf.reshape(t, [-1, 1]), dtype=tf.float64)
        t0 = tf.convert_to_tensor(tf.reshape(tf.convert_to_tensor(t0,dtype=tf.float64), shape=[-1, 1]), dtype=tf.float64)
        x0 = tf.reshape(tf.convert_to_tensor(x0, dtype=tf.float64), shape=[-1, 1])
        x_0 = tf.reshape(tf.convert_to_tensor(x_0, dtype=tf.float64), shape=[-1, 1])
        w_geo = tf.constant(w_geo, dtype=tf.float64)
        model = pinn(
            width=width,
            depth=depth,
            in_dim=in_dim,
            out_dim=out_dim,
            activation=activation,
            optimizer=optimizer,
            lr=lr,
            x0=x0,
            x_0=x_0,
            t0=t0,
            c =c,
            k=k,
            m=m,
            l=l,
            u=u,
            w_geo=w_geo,
            random_seed=random_seed,
        )
        model.compile()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        losses = []
        energy_errors = []
        p = tf.ones_like(t,dtype=tf.float64) * x_0
        q = tf.ones_like(t,dtype=tf.float64) * x0 
        for epoch in range(EPOCHS):
            loss = train_step(model, optimizer, t,p,q)
            losses.append(loss)
            if loss == np.min(losses[-100:]):
                output = model.predict(t,p,q)[:-1,:]
                p,q=  output[:,0],output[:,1]
                p = tf.reshape(p,[-1,1])
                q = tf.reshape(q,[-1,1])
                p = tf.concat([x_0,p],axis=0)
                q = tf.concat([x0,q],axis=0)
            end_time = tf.reshape(u,[-1,1])
            # energy_errors.append(calc_energy_error(model,end_time,x0,x_0,k,m))
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}: Loss: {loss}")

        # plot_energy_error(path, loss_filename, energy_errors, end_time, collocation_points, EPOCHS) 
        model.save(os.path.join(os.getcwd(), model_filename))
        return model
    
if __name__ == "__main__":
    main(*params())
