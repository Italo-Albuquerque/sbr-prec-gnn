import matplotlib.pyplot as plt
import numpy as np

# Dados extraídos diretamente da Tabela 4 do seu PDF (Tempos Uniformes)
# e da Tabela 7 (Tempos Locais)
dados = {
    '10':  {'uniforme': 0.553, 'local': 0.577},
    '30':  {'uniforme': 1.373, 'local': 1.394},  # Onde a IA apontou o erro
    '45':  {'uniforme': 1.553, 'local': 1.553},  # Usando k=2 para bater com a local
    '51':  {'uniforme': 3.606, 'local': 3.606},
    '54':  {'uniforme': 6.419, 'local': 6.425},  # Usando k=0
    '57':  {'uniforme': 1.782, 'local': 1.874},  # Usando k=0
    '61':  {'uniforme': 3.554, 'local': 3.648},  # Usando k=0
    '67':  {'uniforme': 1.702, 'local': 1.836},  # Usando k=0
    '68':  {'uniforme': 5.725, 'local': 5.741},  # Usando k=0
    '69':  {'uniforme': 2.103, 'local': 2.129},  # Usando k=1
    '76':  {'uniforme': 2.712, 'local': 2.712},  # Usando k=2
    '89':  {'uniforme': 5.157, 'local': 5.804},  # Usando k=0
    '96':  {'uniforme': 5.992, 'local': 5.992},  # Usando k=2
    '98':  {'uniforme': 2.898, 'local': 4.706},  # Usando k=0
    '99':  {'uniforme': 1.985, 'local': 3.810},  # Usando k=0
    '105': {'uniforme': 2.193, 'local': 4.628},  # Usando k=0
}

# Extraindo os dados para plotar
matrizes = list(dados.keys())
best_uniform = [dados[m]['uniforme'] for m in matrizes]
best_local = [dados[m]['local'] for m in matrizes]

# Configurando o gráfico
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig, ax = plt.subplots(figsize=(12, 4.5))
x = np.arange(len(matrizes))
width = 0.35

ax.bar(x - width / 2, best_uniform, width,
       label="Melhor configuração uniforme", color="#8c8c8c",
       edgecolor="black", linewidth=0.5)
ax.bar(x + width / 2, best_local, width,
       label="Melhor configuração adaptativa", color="#2ca02c",
       edgecolor="black", linewidth=0.5)

# Abre espaco no topo para a legenda
ymax = max(max(best_uniform), max(best_local)) * 1.30
ax.set_ylim(0, ymax)

ax.set_xlabel("Matriz")
ax.set_ylabel("Tempo em segundos")
ax.set_title("Comparativo entre a melhor configuração uniforme e a adaptativa")
ax.set_xticks(x)
ax.set_xticklabels(matrizes, rotation=45, ha="right")
ax.legend(loc="upper left", framealpha=0.95)
ax.grid(axis="y", alpha=0.3)

plt.savefig("figuras_spe10/ganho_adaptativo.png")
print("Gráfico gerado com sucesso em figuras_spe10/ganho_adaptativo.png")
