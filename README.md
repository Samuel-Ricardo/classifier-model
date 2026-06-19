# Classifier Model - Projeto 04

Projeto academico e experimental com MindSpore para classificacao, cobrindo exemplos basicos de tensores, MLP para MNIST e treino com arquiteturas mais robustas.

## Objetivo

Demonstrar um fluxo progressivo:

- operacoes iniciais de MindSpore (tensores e operadores);
- rede simples para classificacao;
- pipeline de dados do MNIST;
- scripts de treino e inferencia para classificacao de imagens.

## Estrutura

```text
classifier-model/
├── README.md
├── requirements.txt
├── pyproject.toml
├── main.py
├── src/
│   ├── model.py
│   ├── mnist_mlp.py
│   ├── resp.py
│   └── function/
│       ├── activation.py
│       └── loss.md
├── MNIST_Data/
├── docs/
├── v-classifier-model/
└── vcm/
```

## Requisitos

- Python 3.11 recomendado
- MindSpore 2.8.0 (conforme requirements.txt)
- numpy, download, ipykernel

## Setup

No PowerShell, dentro de 04/classifier-model:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Alternativa com uv:

```powershell
uv venv .venv311 --python 3.11
.\.venv311\Scripts\Activate.ps1
uv pip install -r requirements.txt -i https://pypi.org/simple
```

## Como executar

Script principal de exploracao:

```powershell
python main.py
```

Exemplo direto de operacoes e carga de dataset:

```powershell
python src/model.py
```

## Datasets

- MNIST_Data (treino/teste)
- o script tenta localizar o dataset em caminhos comuns e pode baixar automaticamente quando necessario

## Observacoes

- o pyproject.toml ainda esta enxuto e pode ser expandido com metadata e dependencias formais;
- existem dois ambientes virtuais antigos no diretorio; considerar consolidar para um unico ambiente.

## Arquivos-chave

- main.py
- src/model.py
- src/mnist_mlp.py
- src/resp.py
