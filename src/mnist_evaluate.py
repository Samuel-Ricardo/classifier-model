# =============================================================================
# ES510 - Introdução à IA | UFPE | Prof. David Guerra
# Laboratório 4 - Seção 6.3: Avaliação do Modelo
#
# Objetivo: Implementar a função de teste para avaliar o modelo treinado
# no conjunto de teste do MNIST, calculando a acurácia final.
# =============================================================================

#%%
# ---------------------------------------------------------------------------
# ETAPA 1 – Importar bibliotecas necessárias
#
# Reutilizamos as mesmas bibliotecas da seção 6.2 (mnist_mlp.py).
# "mindspore" é o framework principal.
# "numpy" será usado para converter tensores em arrays manipuláveis.
# ---------------------------------------------------------------------------

import mindspore
import numpy as np
from mindspore import nn
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset


#%%
# ---------------------------------------------------------------------------
# ETAPA 2 – Baixar o dataset MNIST
#
# O dataset MNIST contém:
#   - 60.000 imagens de treino
#   - 10.000 imagens de teste
# Cada imagem é um dígito manuscrito (0 a 9) de 28x28 pixels em escala de cinza.
# ---------------------------------------------------------------------------

from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"

path = download(url, "./", kind="zip", replace=True)


#%%
# ---------------------------------------------------------------------------
# ETAPA 3 – Preparar os datasets de treino e TESTE
#
# Diferente de mnist_mlp.py, aqui também preparamos o test_dataset,
# pois ele será usado na avaliação (seção 6.3).
#
# Transformações aplicadas:
#   - Rescale: normaliza pixels de [0, 255] para [0.0, 1.0]
#   - HWC2CHW: reorganiza dimensões de (H, W, C) para (C, H, W)
#   - TypeCast: converte os rótulos para int32 (exigido pela função de perda)
# ---------------------------------------------------------------------------

train_dataset = MnistDataset('MNIST_Data/train')
test_dataset  = MnistDataset('MNIST_Data/test')   # <- usado na avaliação

image_transforms = [
    vision.Rescale(1.0 / 255.0, 0.0),  # normalização dos pixels
    vision.HWC2CHW()                    # reordena dimensões da imagem
]

label_transform = transforms.TypeCast(mindspore.int32)  # converte tipo dos rótulos

# Aplica transformações ao dataset de treino
train_dataset = train_dataset.map(operations=image_transforms, input_columns="image")
train_dataset = train_dataset.map(operations=label_transform,  input_columns="label")
train_dataset = train_dataset.batch(32)

# Aplica as MESMAS transformações ao dataset de TESTE
# Isso garante que os dados de teste estejam no mesmo formato que os de treino
test_dataset = test_dataset.map(operations=image_transforms, input_columns="image")
test_dataset = test_dataset.map(operations=label_transform,  input_columns="label")
test_dataset = test_dataset.batch(32)


#%%
# ---------------------------------------------------------------------------
# ETAPA 4 – Definir a arquitetura da rede neural (Network)
#
# Mesma arquitetura usada em mnist_mlp.py:
#   - Dense(784 -> 512): camada de entrada (28x28 = 784 pixels)
#   - ReLU: função de ativação
#   - Dense(512 -> 512): camada oculta
#   - ReLU: função de ativação
#   - Dropout(0.5): regularização para evitar overfitting
#   - Dense(512 -> 10): saída com 10 classes (dígitos 0-9)
# ---------------------------------------------------------------------------

class Network(nn.Cell):

    def __init__(self):
        super().__init__()

        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28 * 28, 512),    # entrada: 784 pixels achatados
            nn.ReLU(),                  # ativação não-linear
            nn.Dense(512, 512),         # camada oculta
            nn.ReLU(),                  # ativação não-linear
            nn.Dropout(keep_prob=0.5),  # durante treino, desativa 50% dos neurônios
            nn.Dense(512, 10)           # saída: 10 classes (0 a 9)
        )

    def construct(self, x):
        # reshape: transforma imagem (batch, 1, 28, 28) em vetor (batch, 784)
        x = x.reshape(x.shape[0], -1)
        logits = self.dense_relu_sequential(x)
        return logits  # retorna logits (valores brutos, sem softmax)


#%%
# ---------------------------------------------------------------------------
# ETAPA 5 – Instanciar modelo, função de perda e otimizador
#
# - SoftmaxCrossEntropyWithLogits: função de perda para classificação multi-classe
#   (aplica softmax internamente antes de calcular a entropia cruzada)
# - SGD: otimizador de gradiente descendente estocástico
#   - learning_rate: tamanho do passo no gradiente
#   - momentum: acelera convergência aproveitando direção dos gradientes anteriores
#   - weight_decay: regularização L2 (penaliza pesos grandes)
# ---------------------------------------------------------------------------

model = Network()

loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

optimizer = nn.SGD(
    model.trainable_params(),
    learning_rate=1e-2,
    momentum=0.9,
    weight_decay=0.0001
)


#%%
# ---------------------------------------------------------------------------
# ETAPA 6 – Definir função forward e cálculo de gradiente
#
# forward_fn: executa a passagem direta (forward pass) e calcula a perda.
# value_and_grad: cria uma função que retorna tanto o valor da perda
#                 quanto os gradientes em relação aos parâmetros do otimizador.
# has_aux=True: indica que forward_fn retorna valores auxiliares (logits)
#               além da perda principal.
# ---------------------------------------------------------------------------

def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits  # retorna perda E logits (aux)

grad_fn = mindspore.value_and_grad(
    forward_fn,
    None,
    optimizer.parameters,
    has_aux=True
)


#%%
# ---------------------------------------------------------------------------
# ETAPA 7 – Definir passo de treinamento e função de treino
#
# train_step: executa um passo de otimização para um batch.
#   1. Calcula perda e gradientes via grad_fn
#   2. Atualiza parâmetros com optimizer(grads)
#   3. Retorna o valor da perda para monitoramento
#
# train: itera sobre todos os batches do dataset para uma época.
#   - model.set_train(): ativa modo de treino (Dropout funciona normalmente)
#   - Exibe perda a cada 100 batches para acompanhar a convergência
# ---------------------------------------------------------------------------

def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)  # atualiza os pesos do modelo
    return loss

def train(model, dataset):
    size = dataset.get_dataset_size()
    model.set_train()  # ativa modo de treinamento

    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")


#%%
# ---------------------------------------------------------------------------
# ETAPA 8 – Treinar o modelo por 3 épocas
#
# Uma época = o modelo vê todo o dataset de treino uma vez.
# 3 épocas é suficiente para demonstração; mais épocas = mais acurácia.
# ---------------------------------------------------------------------------

epochs = 3

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(model, train_dataset)

print("Done!\n")


#%%
# ---------------------------------------------------------------------------
# ETAPA 9 (SEÇÃO 6.3) – Implementar a função de avaliação (EXERCÍCIO)
#
# Objetivo: medir o desempenho do modelo em dados NUNCA vistos no treino.
#
# Como funciona:
#   1. model.set_train(False): desativa o Dropout (modo avaliação)
#   2. Para cada batch do test_dataset:
#      a) Passa os dados pelo modelo -> obtém logits (vetor com 10 valores)
#      b) argmax(logits): encontra o índice do maior valor = classe prevista
#      c) Compara com o rótulo real (label)
#      d) Conta acertos (predicted == label)
#   3. Acurácia = total de acertos / total de amostras
#
# Por que argmax?
#   A rede retorna um vetor de 10 valores (um por classe).
#   O maior valor indica a classe com maior probabilidade prevista.
#   Exemplo: logits = [0.1, 0.05, 0.8, ...] -> argmax = 2 -> classe "2"
# ---------------------------------------------------------------------------

def test(model, dataset):
    """
    Avalia o modelo no conjunto de teste.

    Parâmetros:
        model   : rede neural treinada (instância de Network)
        dataset : dataset de teste já transformado e em batches

    Retorna:
        accuracy (float): porcentagem de acertos no conjunto de teste
    """

    # Desativa o modo de treinamento
    # Com set_train(False), o Dropout fica inativo e todos os neurônios contribuem
    model.set_train(False)

    total   = 0  # número total de amostras avaliadas
    correct = 0  # número de acertos

    for data, label in dataset.create_tuple_iterator():

        # Passagem direta: obtém os logits para cada amostra do batch
        # logits.shape = (batch_size, 10) — 10 valores por imagem
        logits = model(data)

        # argmax retorna o índice do maior valor em cada linha
        # Esse índice é a classe prevista pelo modelo (0 a 9)
        predicted = logits.argmax(axis=1)  # shape: (batch_size,)

        # Converte tensores para numpy para facilitar comparação
        predicted = predicted.asnumpy()
        label     = label.asnumpy()

        # Conta quantas previsões foram corretas neste batch
        correct += (predicted == label).sum()
        total   += label.shape[0]

    # Acurácia = número de acertos / total de amostras
    accuracy = correct / total
    return accuracy


#%%
# ---------------------------------------------------------------------------
# ETAPA 10 – Executar a avaliação e exibir o resultado
#
# Chamamos a função test() com o test_dataset (10.000 imagens).
# Uma boa acurácia esperada após 3 épocas: ~95% a ~97%.
# ---------------------------------------------------------------------------

print("Avaliando o modelo no conjunto de teste...")
accuracy = test(model, test_dataset)

print(f"\nResultado da Avaliação:")
print(f"  Acurácia no conjunto de teste: {accuracy * 100:.2f}%")
print(f"  ({int(accuracy * 10000)}/10000 imagens classificadas corretamente)\n")


#%%
# ---------------------------------------------------------------------------
# DESAFIO ADICIONAL – Exibir amostras do teste com classe real vs. prevista
#
# Objetivo: visualizar individualmente algumas predições do modelo.
#
# Para cada imagem exibida:
#   - "Real"    : rótulo correto fornecido pelo dataset
#   - "Previsto": classe que o modelo previu
#   - Se coincidem -> CORRETO, senão -> ERRADO
# ---------------------------------------------------------------------------

print("=" * 50)
print("DESAFIO: Amostras do conjunto de teste")
print("=" * 50)

model.set_train(False)  # garante modo de avaliação

# Pega apenas o primeiro batch para demonstração
for data, label in test_dataset.create_tuple_iterator():

    logits    = model(data)                # saída da rede (logits)
    predicted = logits.argmax(axis=1)      # classe prevista (argmax)

    predicted = predicted.asnumpy()
    label     = label.asnumpy()

    # Exibe as primeiras 10 amostras do batch
    print(f"\n{'Amostra':<10} {'Real':<10} {'Previsto':<12} {'Resultado'}")
    print("-" * 45)

    for i in range(10):
        real  = label[i]
        pred  = predicted[i]
        check = "✓ CORRETO" if real == pred else "✗ ERRADO"
        print(f"{i + 1:<10} {real:<10} {pred:<12} {check}")

    break  # exibe apenas o primeiro batch

print("\n" + "=" * 50)
print("Avaliação concluída!")
print("=" * 50)
