#%%

import mindspore as ms 

from mindspore import nn
from mindspore.train import Model
from mindspore.train.callback import LossMonitor

import mindspore.dataset as ds 
from mindspore.nn import Accuracy


#%%

sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU()
prelu = nn.PReLU()
gelu = nn.GELU()


#%%

from mindspore import Tensor
import numpy as np 

#%%

x = Tensor(np.array([-2.0,-1.0, 0.0, 1.0, 2.0]), dtype=ms.float32)

print("Sigmoid:", sigmoid(x))
print("Tanh:", tanh(x))
print("ReLU:", relu(x))
print("Leaky ReLU:", leaky_relu(x))
print("PReLU:", prelu(x))
print("GeLU:", gelu(x))

#%%

print("Resposta: Tanh mantém valores negativos em [-1, 1]; ReLU zera os negativos.")


#%%


#%%

