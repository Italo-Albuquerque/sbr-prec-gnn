# Redes Neurais em Grafos Aplicadas ao Pré-Condicionamento Adaptativo de Sistemas Lineares Esparsos

Este repositório contém o código-fonte desenvolvido para a qualificação de mestrado no Programa de Pós-Graduação em Modelagem e Métodos Quantitativos da UFC.

- **Autor:** Ítalo Albuquerque da Silva
- **Orientador:** Prof. Dr. Albert Einstein Fernandes Muritiba
- **Coorientador:** Prof. Dr. Michael Ferreira de Souza

## 📌 Sobre o Projeto

O trabalho investiga a seleção adaptativa e local do nível de preenchimento $k$ em pré-condicionadores ILU($k$) para o método GMRES. O objetivo é usar Redes Neurais em Grafos (GNNs) para prever o melhor $k$ para cada subdomínio em ambientes de computação paralela com decomposição de domínios (Block Jacobi), minimizando o tempo de solução de sistemas lineares de grande porte oriundos de simulações de reservatórios de petróleo (CMG IMEX).

O projeto foi dividido em duas fases:

- **Fase 1 (`Fase_1_mxspe001`):** prova de conceito com matrizes de pequeno porte ($n \approx 300$). Utilização de um modelo GraphSAGE.
- **Fase 2 (`Fase_2_SPE10`):** expansão para matrizes industriais ($n = 1\,094\,422$). Extração de 16 atributos de nós e 4 de arestas, agrupamento espacial dos subdomínios, geração de rótulos suaves via modelo *Teacher* e treinamento de modelos preditivos (Random Forest, MLP, GraphSAGE e GINE).

## ⚙️ Dependências

Os códigos foram desenvolvidos em Python 3.14 no ambiente Linux (WSL2). As principais bibliotecas utilizadas estão listadas no arquivo `requirements.txt`. Para instalá-las, rode:

```bash
pip install -r requirements.txt
```
