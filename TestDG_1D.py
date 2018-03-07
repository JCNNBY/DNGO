import numpy as np
from pyDOE import *
import torch
from torch.autograd import Variable
from SimpleNeuralNet import Net
from DNGO import DG
import matplotlib.pyplot as plt

def init_random_uniform(lower, upper, n_points, rng=None):
    """
    Samples N data points uniformly.

    Parameters
    ----------
    lower: np.ndarray (D)
        Lower bounds of the input space
    upper: np.ndarray (D)
        Upper bounds of the input space
    n_points: int
        The number of initial data points
    rng: np.random.RandomState
            Random number generator
    Returns
    -------
    np.ndarray(N,D)
        The initial design data points
    """

    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    n_dims = lower.shape[0]

    return np.array([rng.uniform(lower, upper, n_dims) for _ in range(n_points)])

def Sampling(Interval, n, constraint = None):
	# n is the number of samples
	N_cst = []
	N_trt = []
	dim = Interval.shape[0]
	sample = lhs(dim, samples=n, criterion="c")
	for i in range(dim):
		length = Interval[i,1] - Interval[i,0]
		for j in range(n):
			sample[j,i] = Interval[i,0] + (length * sample[j,i])
	return sample
    
def f(x):
    return np.sinc(x * 10 - 5).sum(axis=1)[:, None]
    
N, D_in, H, D, D_out = 30, 1, 50, 10, 1
epoch = 1000
inter = np.zeros([D_in, 2])
inter[0,0] = 0.0
inter[0,1] = 1.0
rng = np.random.RandomState(42)
#DOE = init_random_uniform(np.zeros(1), np.ones(1), 30, rng)
DOE = Sampling(inter, N)
xtest = np.linspace(0, 1, 100)[:, None]
yvalues = f(DOE)[:,0]

x = Variable(torch.from_numpy(DOE).float())
x_test = Variable(torch.from_numpy(xtest).float())
y = Variable(torch.from_numpy(yvalues).float(), requires_grad=False)
fvals = f(xtest)

deepgaussian = DG(10000, 1e-2, H, D)
deepgaussian.train(DOE,yvalues)
mean, var = deepgaussian.predict(x_test)
plt.plot(DOE, yvalues, "ro")
plt.plot(xtest[:,0], fvals, "k--")
plt.plot(xtest[:,0], mean, "blue")
plt.fill_between(xtest[:, 0], mean + np.sqrt(var), mean - np.sqrt(var), color="orange", alpha=0.4)
plt.grid()
plt.show()











