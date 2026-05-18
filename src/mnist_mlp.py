#%%

import mindspore

from mindspore import nn
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset


#%%

from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
"notebook/datasets/MNIST_Data.zip"

path = download(url, "./", kind="zip",replace=True) 


#%%

train_dataset= MnistDataset('MNIST_Data/train')
test_dataset= MnistDataset('MNIST_Data/test')

image_transforms = [
    vision.Rescale(1.0 / 255.0, 0.0),
    vision.HWC2CHW()
]

label_transform = transforms.TypeCast(mindspore.int32)

train_dataset= train_dataset.map(operations=image_transforms, input_columns="image")
train_dataset= train_dataset.map(operations=label_transform, input_columns="label")

train_dataset = train_dataset.batch(32)


#%%

class Network(nn.Cell):

    def __init__(self):
        super().__init__()

        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(512, 10)
        )

    def construct(self, x):
    
        x = x.reshape(x.shape[0], -1)
        logits = self.dense_relu_sequential(x)
        return logits

#%%

model = Network()

loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

optimizer = nn.SGD(
    model.trainable_params(),
    learning_rate=1e-2,
    momentum=0.9,
    weight_decay=0.0001
)

#%%

def foward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits

#%%

grad_fn = mindspore.value_and_grad(
    foward_fn,
    None,
    optimizer.parameters,
    has_aux=True
)


#%%

"""
def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss
"""

#%%

def train_step(data, label):
    (loss, _), grade = grad_fn(data, label)
    optimizer(grade)
    return loss

def train(model, dataset):
    size = dataset.get_dataset_size()
    model.set_train()

    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss:{loss:>7f} [{current:>3d}/{size:>3d}]")


#%%

epochs = 3

for t in range(epochs):
    print(f"Epoch{t+1}\n-------------------------------")
    train(model, train_dataset)

print("Done!")
