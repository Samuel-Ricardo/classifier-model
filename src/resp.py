#%%
import mindspore
import numpy as np
from mindspore import nn
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
#%%


#%%
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"

path = download(url, "./", kind="zip", replace=True)
#%%


#%%
train_dataset = MnistDataset('MNIST_Data/train')
test_dataset = MnistDataset('MNIST_Data/test')
#%%


#%%
image_transforms = [
    vision.Rescale(1.0 / 255.0, 0.0),
    vision.HWC2CHW()
]

label_transform = transforms.TypeCast(mindspore.int32)
#%%


#%%
train_dataset = train_dataset.map(operations=image_transforms, input_columns="image")
train_dataset = train_dataset.map(operations=label_transform,  input_columns="label")
train_dataset = train_dataset.batch(32)
#%%

#%%
test_dataset = test_dataset.map(operations=image_transforms, input_columns="image")
test_dataset = test_dataset.map(operations=label_transform, input_columns="label")
test_dataset = test_dataset.batch(32)
#%%


#%%
class Network(nn.Cell):

    def __init__(self):
        super().__init__()

        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28 * 28, 512),
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


#%%

model = Network()

loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

optimizer = nn.SGD(
    model.trainable_params(),
    learning_rate=1e-2,
    momentum=0.9,
    weight_decay=0.0001 
)
#%%


#%%
def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits 

grad_fn = mindspore.value_and_grad(
    forward_fn,
    None,
    optimizer.parameters,
    has_aux=True
)
#%%

#%%
def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss

def train(model, dataset):
    size = dataset.get_dataset_size()
    model.set_train()

    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f} [{current:>3d}/{size:>3d}]")
#%%


#%%
epochs = 3 

for t in range(epochs): 
    print(f"Epoch {t+1}\n-------------------------------")
    train(model, train_dataset)

print("Done!\n")
#%%


#%%
def test(model, dataset):
    model.set_train(False)

    total   = 0 
    correct = 0 

    for data, label in dataset.create_tuple_iterator():
        
        logits = model(data)

        predicted = logits.argmax(axis=1)
        
        predicted = predicted.asnumpy()
        label     = label.asnumpy()

        correct += (predicted == label).sum()
        total   += label.shape[0]

    accuracy = correct / total
    return accuracy
#%%


#%%
print("Avaliando o modelo no conjunto de teste...")
accuracy = test(model, test_dataset)

print(f"\nResultado da avaliação: \n")
print(f"Acurácia = {accuracy*100:.2f}%")
print(f"  ({int(accuracy * 10000)}/10000 imagens classificadas corretamente)\n")
#%%


#%%
print("=" * 50)
print("DESAFIO: Amostras do conjunto de teste")
print("=" * 50)
#%%

#%%
model.set_train(False)

for data, label in test_dataset.create_tuple_iterator():

    logits    = model(data)
    predicted = logits.argmax(axis=1)

    predicted = predicted.asnumpy()
    label     = label.asnumpy()

    print(f"\n{'Amostra':<10} {'Real':<10} {'Previsto':<12} {'Resultado'}")
    print("-" * 45)

    for i in range(10):
        real = label[i]
        pred = predicted[i]
        check = "CORRETO" if real == pred else "ERRADO"
        print(f"{i + 1:<10} {real:<10} {pred:<12} {check}")
    break
#%%

#%%
print("\n" + "=" * 50)
print("Avaliação concluída!")
print("=" * 50)
#%%

