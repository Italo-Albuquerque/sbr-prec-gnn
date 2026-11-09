import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os

# Configuração estética
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Cria a pasta para salvar as figuras (se não existir)
os.makedirs("figuras_resultados", exist_ok=True)

df = pd.read_csv("ilu_all_results.csv")
df_mat10 = df[df["matrix_id"] == 10]

# Média e desvio padrão das iterações
mean_its = df_mat10.groupby("k")["its"].mean().reset_index()
std_its = df_mat10.groupby("k")["its"].std().reset_index()

plt.figure(figsize=(6,4))
sns.barplot(data=mean_its, x="k", y="its", hue="k", legend=False, palette="Blues_d")
plt.errorbar(x=mean_its["k"], y=mean_its["its"], yerr=std_its["its"], 
             fmt='none', ecolor='black', capsize=5, elinewidth=1.5)
plt.title("Número médio de iterações do GMRES para diferentes níveis de ILU(k)\n(Matriz 10)")
plt.xlabel("Nível de preenchimento (k)")
plt.ylabel("Iterações")
plt.ylim(0, max(mean_its["its"])*1.1)
plt.savefig("figuras_resultados/fig_iteracoes_vs_k.pdf", bbox_inches="tight")
plt.close()

mean_time = df_mat10.groupby("k")["time"].mean().reset_index()
std_time = df_mat10.groupby("k")["time"].std().reset_index()

plt.figure(figsize=(6,4))
sns.barplot(data=mean_time, x="k", y="time", hue="k", legend=False, palette="Reds_d")
plt.errorbar(x=mean_time["k"], y=mean_time["time"], yerr=std_time["time"],
             fmt='none', ecolor='black', capsize=5, elinewidth=1.5)
plt.title("Tempo médio de solução (fatoração + GMRES) para diferentes k\n(Matriz 10)")
plt.xlabel("Nível de preenchimento (k)")
plt.ylabel("Tempo (s)")
plt.savefig("figuras_resultados/fig_tempo_vs_k.pdf", bbox_inches="tight")
plt.close()

if not os.path.exists("dataset.csv"):
    print("Arquivo dataset.csv não encontrado. Execute build_dataset.py primeiro.")
else:
    df_dataset = pd.read_csv("dataset.csv")
    counts = df_dataset["best_k"].value_counts().sort_index()
    plt.figure(figsize=(5,4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, palette="viridis")
    plt.title("Distribuição do melhor k global por matriz\n(critério: tempo total)")
    plt.xlabel("Melhor k")
    plt.ylabel("Número de matrizes")
    for i, v in enumerate(counts.values):
        plt.text(i, v+0.2, str(v), ha='center')
    plt.savefig("figuras_resultados/fig_distribuicao_best_k.pdf", bbox_inches="tight")
    plt.close()

try:
    labels = torch.load("labels_10_p2.pt", weights_only=False).numpy()
    subdomains = [0, 1]
    plt.figure(figsize=(4,3))
    plt.bar(subdomains, labels, color=['blue', 'orange'])
    plt.xticks(subdomains, ["Subdomínio 0", "Subdomínio 1"])
    plt.title("Melhor k por subdomínio\n(Matriz 10, 2 processos MPI)")
    plt.xlabel("Subdomínio")
    plt.ylabel("k ótimo")
    plt.ylim(0, 3.5)
    for i, v in enumerate(labels):
        plt.text(i, v+0.1, str(int(v)), ha='center')
    plt.savefig("figuras_resultados/fig_best_k_por_subdominio.pdf", bbox_inches="tight")
    plt.close()
except FileNotFoundError:
    print("Arquivo labels_10_p2.pt não encontrado. Execute label_subdomains.py primeiro.")
    print("Comando: mpiexec -n 2 python label_subdomains.py mat_10.txt")

print("Todos os gráficos disponíveis foram gerados na pasta 'figuras_resultados'.")
