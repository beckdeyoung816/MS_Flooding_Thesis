# %%
import numpy as np

# %%
# Loss function based on the gumbel distribution
def gumbel_loss(y_true, y_pred, gamma=1):
    delta = y_pred - y_true
    return np.mean((1-np.exp(-delta**2)) ** (gamma) * delta**2)
    

def frechet_loss(y_true, y_pred, alpha=1, s=3):
    delta = y_pred - y_true
    
    delta_S = (delta + s*(alpha/(1+alpha) ** (1/alpha))) / s

    loss = (-1-alpha) * (-delta_S) ** (-alpha) + \
        np.log(delta_S)
    
    return np.mean(loss)

    
    
# %%
# Make an array of random numbers of length 10
y_true = np.random.rand(10)
y_pred = np.random.rand(10)

delta = y_pred - y_true
# %%
