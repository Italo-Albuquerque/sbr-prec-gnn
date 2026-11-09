import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

#Carrega os dados reais da Fase 1
df = pd.read_csv("ilu_all_results.csv")

#Filtra apenas as execuções que convergiram
df_ok = df[df["converged"] == True]

#Calcula o tempo médio para cada matriz e cada k
mean_times = df_ok.groupby(["matrix_id", "k"])["time"].mean().reset_index()

#Encontra o k com o menor tempo médio para cada matriz
best_k_per_matrix = mean_times.loc[mean_times.groupby("matrix_id")["time"].idxmin()]

#Conta quantas vezes cada k foi o vencedor
contagem = best_k_per_matrix["k"].value_counts().sort_index()

# Garante que todos os k's (0, 1, 2, 3) apareçam no eixo
valores = [contagem.get(k, 0) for k in [0, 1, 2, 3]]

#Plota o gráfico
fig, ax = plt.subplots(figsize=(6, 4))
cores = ['#440154', '#21918c', '#fde725', '#d62728']
bars = ax.bar([str(k) for k in [0, 1, 2, 3]], valores, color=cores, edgecolor='black', linewidth=0.5)

#Adiciona os números em cima das barras
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, int(yval), ha='center', va='bottom', fontsize=11)

ax.set_xlabel("Melhor k")
ax.set_ylabel("Número de matrizes")
ax.set_title("Distribuição do melhor k global por matriz\n(critério: tempo total)")
ax.set_ylim(0, max(valores) + 2)
ax.grid(axis='y', alpha=0.3)

#Salva a imagem na pasta correta
output_dir = "figuras_resultados"
os.makedirs(output_dir, exist_ok=True)
caminho_saida = os.path.join(output_dir, "fig_distribuicao_best_k_correta.png")

plt.savefig(caminho_saida)
print(f"Figura gerada com sucesso em: {caminho_saida}")
print("Contagem de vitórias por k:", dict(zip([0, 1, 2, 3], valores)))
