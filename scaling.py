import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.autograd import Variable



class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.multp = Variable(torch.rand(1), requires_grad=True)

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.multp = nn.Parameter(torch.rand(1)) # requires_grad is True by default for Parameter

m1 = Model1()
m2 = Model2()

print('m1', list(m1.parameters()))
print('m2', list(m2.parameters()))
