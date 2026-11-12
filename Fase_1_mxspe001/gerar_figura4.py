import matplotlib.pyplot as plt
import numpy as np

# Configurações de fonte e tamanho
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Dados corretos (conforme a Tabela 2 do texto)
subdominios = ['Subdomínio 0', 'Subdomínio 1']
ks = [1, 3] # k0=1, k1=3

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(subdominios, ks, color=['blue', 'blue'], width=0.5)

# Adiciona os números em cima das barras
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom', fontsize=12)

# Rótulos e título
ax.set_xlabel("Subdomínio")
ax.set_ylabel("k ótimo")
ax.set_title("Melhor k por subdomínio\n(Matriz 10, 2 processos MPI)")
ax.set_ylim(0, 4) # Limite do eixo y para dar espaço
ax.grid(axis='y', alpha=0.3)

# Salva a imagem na pasta atual
plt.savefig("fig_best_k_por_subdominio_corrigida.png")
print("Figura gerada com sucesso: fig_best_k_por_subdominio_corrigida.png")
