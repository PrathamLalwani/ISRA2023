﻿### Training for 10,0000 EPOCHS
#### k = 20.
#### m = 1.
#### collocation_points = 256
#### t $\in$ [0,3]
$$H(p,q) = \frac{p^2}{2m} + \frac{1}{2} kq^2$$

Consider $\hat{p}$ and $\hat{q}$ to be outputs from the neural network and $\hat{H} = H(\hat{p},\hat{q})$
#### PINN
$$\mathcal{L}_{PINN} = \mathcal{L}_{boundary} + \frac{1}{N} \sum_{i=1}^N |f(t_i)|^2$$ 
#### PINN without Hamiltonian residual
$$\mathcal{L}_{PINN-geo} = \mathcal{L}_{boundary} + \frac{1}{N} \sum_{i=1}^N |f(t_i)|^2 + \frac{1}{N} \sum_{i=1}^N \left|\frac{(H-H_0)}{H_0}\right|^2$$
#### PINN with Hamiltonian residual
$$ \mathcal{L}_{PINN-geo-hamiltonian-residual} = \mathcal{L}_{boundary} 
+ \frac{1}{N} \sum_{i=1}^N |\frac{\partial{\hat{H}}}{\partial{\hat{q}}} - \frac{d\hat{p}}{dt}|^2  
+ \frac{1}{N} \sum_{i=1}^N |\frac{\partial{\hat{H}}}{\partial{\hat{p}}} +\frac{d\hat{q}}{dt}|^2  
+ \frac{1}{N} \sum_{i=1}^N \left|\frac{(\hat{H}-H_0)}{H_0}\right|^2 $$

Activation function: tanh

Optimizer: Adam

Learning rate: 0.001

Activation function: SIREN - sin(x) as activation function, with the weights constrained to be in the range $[-6/\sqrt(d_{in}), 6/\sqrt(d_{in})]$ where $d_{in}$ is the number of input units.
