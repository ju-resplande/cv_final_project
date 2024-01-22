## Computer Vision 2023/2 - Final project: FGSM attacks

Este repositório contém o trabalho final durante o curso de Visão Computacional, ofertado pelo Instituto de Informática da Universidade Federal de Goiás, ministrado pelo Prof. Gustavo Teodoro Laureano no semestre de 2023/2.

O presente projeto possui o seguinte escopo:
- **Tema**: Ataque adversário em pedra, papel e tesoura.
- **Técnica**: Fast Gradient Signed Method (FGSM).
- **Desafio**: Não foram encontradas implementações de treino.

## Instalação
### Dependências

- Python 3
- Conda. (Opcional para criação e gerenciamento de ambiente)

### Requerimentos

```bash
pip install -r requirements.txt
```

## Artefatos
### Notebooks
- [notebooks/example.ipynb](notebooks/example.ipynb): Exemplo de geração de imagem adversária
- [notebooks/stats.ipynb](notebooks/stats.ipynb): Estaísticas e resultados gerados pós-treino para o relatório e apresentação

### Código
- [config.py](config.py): Definição de parâmetros como `batch_size`, e `optimizer`
- [main.py](main.py): CLI de treino e avaliação
- [model.py](model.py): implementação da classe `ImageClassifierFGSM`, que realiza treino adversário. A classe `ImageClassifierFGSMFramework` é uma implementação alto nível, em que acrescente funções de preprocessamento, geração de imagens adversárias, treino. predição e avaliação.
- [utils.py](utils.py): Funções utilitárias

### Dados de teste
Os conjuntos de dados de teste estão na pasta [data](data), em que:
- [test_split](data/test_split): conjunto de validação do dataset [rock-paper-scissors](https://laurencemoroney.com/datasets.html)
- [0](data/0): Fotos das mãos de pedra, papel e tesoura do colega 0
- [1](data/1): Fotos das mãos de pedra, papel e tesoura do colega 1
- [2](data/2): Fotos das mãos de pedra, papel e tesoura do colega 2
- [3](data/3): Fotos das mãos de pedra, papel e tesoura do colega 3
