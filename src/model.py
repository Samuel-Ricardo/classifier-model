#%%

import mindspore as ms 
from mindspore import Tensor 
from mindspore import nn
from mindspore.train import Model
from mindspore.train.callback import LossMonitor

import mindspore.dataset as ds 
from mindspore.nn import Accuracy
import numpy as np


#%%

class SimpleMLP(nn.Cell):
    def __init__(self):
        super(SimpleMLP, self).__init__()

        self.fc1 = nn.Dense(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(4, 1)
        self.sigmoid = nn.Sigmoid()


    def construct(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


#%%

X_train = np.array([[1.2,0.7],
                [-0.3,-0.5],
                [3.0, 0.1],
                [-0.1,-1.0],
                [0.5, 0.6],
                [-0.7,-0.8],
                [2.5, 0.4],
                [-0.2,-1.3]], dtype=np.float32)

y_train = np.array([0, 1, 0, 1, 0, 1,0, 1], dtype=np.int32)


#%%

X_train = Tensor(X_train)
y_train = Tensor(y_train.reshape(-1, 1), ms.float32)


#%%


train_dataset = ds.NumpySlicesDataset({"features": X_train, "label": y_train}, shuffle=True)
train_dataset = train_dataset.batch(4)


#%%

net = SimpleMLP()


#%%

criterion = nn.BCELoss(reduction='mean')
optimizer = nn.Adam(net.trainable_params(), learning_rate=0.01)


#%%

model = Model(
    net,
    loss_fn=criterion,
    optimizer=optimizer,
    metrics={"accuracy": Accuracy()}
)


#%%

model.train(
    10,
    train_dataset,
    callbacks=[LossMonitor()],
    dataset_sink_mode=False
)


#%%

net.set_train(False)

X_test = np.array([
    [0.2, 0.1],
    [1.5, 0.3],
    [-0.5,-0.7]
], dtype=np.float32)

X_test = Tensor(X_test)

predictions = model.predict(X_test)
predictions = predictions.asnumpy()

#%%

print("Predicoes (valores continuos): ")
print(predictions)

#%%

predicted_classes = (predictions > 0.5).astype(int)

print("Classes previstas: ")
print(predicted_classes)


#%%


