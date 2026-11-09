import torch
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração estética
sns.set_style("whitegrid")

# Coleta todos os arquivos de grafo e rótulo
data = []
for label_file in glob.glob("labels_*_p2.pt"):
    # Extrai o ID da matriz
    mat_id = label_file.split("_")[1]
    # Carrega os rótulos
    labels = torch.load(label_file, weights_only=False).numpy()
    # Carrega o grafo correspondente
    graph_file = f"graph_{mat_id}_p2.pt"
    if not os.path.exists(graph_file):
        print(f"Aviso: {graph_file} não encontrado. Pulando {mat_id}.")
        continue
    graph = torch.load(graph_file, weights_only=False)
    features = graph.x.numpy()
    # Associa cada nó ao seu rótulo e features
    for i, (feat, label) in enumerate(zip(features, labels)):
        data.append({
            "matrix_id": mat_id,
            "subdomain": i,
            "k": label,
            "n_rows": feat[0],
            "mean_degree": feat[1],
            "density": feat[2]
        })

if not data:
    print("Nenhum dado encontrado. Verifique se os arquivos graph_*_p2.pt e labels_*_p2.pt existem.")
    exit()

df = pd.DataFrame(data)

# Estatísticas descritivas por k
print("=== Estatísticas descritivas por k ===")
print(df.groupby("k")[["n_rows", "mean_degree", "density"]].describe())

print("\n=== Médias por k ===")
print(df.groupby("k")[["n_rows", "mean_degree", "density"]].mean())

# Salva as médias em CSV para gerar tabela LaTeX posteriormente
df.groupby("k")[["n_rows", "mean_degree", "density"]].mean().to_csv("stats_features_means.csv")

# Gera boxplots das features por k
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
features = ["n_rows", "mean_degree", "density"]
titles = ["Tamanho local", "Grau médio", "Densidade média"]
for i, (feat, title) in enumerate(zip(features, titles)):
    sns.boxplot(data=df, x="k", y=feat, ax=axes[i])
    axes[i].set_title(title)
    axes[i].set_xlabel("k")
plt.tight_layout()
plt.savefig("figuras_resultados/boxplot_features.pdf")
plt.close()

print("\nBoxplots salvos em figuras_resultados/boxplot_features.pdf")
