#%% 

import mindspore as ms 
#from mindspore import Tensor 
from mindspore import nn
from mindspore.train import Model
from mindspore.train.callback import LossMonitor

import mindspore.dataset as ds 
from mindspore.nn import Accuracy
#import numpy as np


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

# QUESTAO 4:
#
# A função `Tanh` mantém os valores negativos, apenas os comprime para o intervalo entre $-1$ e $1$. 
# Já a função `ReLU` zera todos os valores negativos e mantém apenas os positivos.

# Pelos resultados, `Tanh(-2)` e `Tanh(-1)` continuam negativos, enquanto `ReLU(-2)` e `ReLU(-1)` viram `0`. 
# Portanto: `Tanh` preserva negativos; `ReLU` os zera.


#%%

