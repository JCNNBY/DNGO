import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
#import numpy as np
#from pyDOE import *

class Net(nn.Module):
    def __init__(self, D_in, H, D, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Net, self).__init__()
        self.inputlayer = nn.Linear(D_in, H)
        self.middle = nn.Linear(H, H)
        self.lasthiddenlayer = nn.Linear(H, D)
        self.outputlayer = nn.Linear(D, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.outputlayer(self.PHI(x))
        return y_pred
    
    def PHI(self, x):
        h_relu = self.inputlayer(x).tanh()
        for i in range(2):
            h_relu = self.middle(h_relu).tanh()
        phi = self.lasthiddenlayer(h_relu)
        return phi




























