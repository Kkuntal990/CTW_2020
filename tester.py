import torch
import tensorflow as tf
import numpy as np


c = torch.tensor([[[ 1, 2, 3],[-1, 1, 4],[1,23,5]], [[ 1, 2, 3],[-1, 1, 4],[1,23,5]]] , dtype= torch.float)
print(c.shape)
#d = torch.tensor([ 1, 2, 3] , dtype= torch.float)

print(torch.norm(c, p=2, dim=[1,2]))

#a = tf.convert_to_tensor([[ 1, 2, 3],[-1, 1, 4],[1,23,5]], dtype=tf.float32)

#print(tf.norm(a, ord='euclidean'))
a = torch.tensor([[1,3,4],[1,3,5]], dtype=torch.float)
b = torch.tensor([[1,2,7],[1,3,8]], dtype=torch.float)
print(torch.norm(a-b, p=2, dim=1))

