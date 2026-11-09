import pandas as pd

df = pd.read_csv("dataset.csv")
# Ordenar por ID da matriz
df_sorted = df.sort_values("matrix_id")

# Gerar código LaTeX
with open("tabela_melhor_k.tex", "w") as f:
    f.write("\\begin{table}[htbp]\n\\centering\n")
    f.write("\\caption{Melhor nível de preenchimento global por matriz (critério tempo)}\n")
    f.write("\\label{tab:melhor_k_global}\n")
    f.write("\\begin{tabular}{c c}\n\\toprule\nMatriz & Melhor k \\\\\n\\midrule\n")
    for _, row in df_sorted.iterrows():
        f.write(f"{int(row['matrix_id'])} & {int(row['best_k'])} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
    f.write("\\end{table}\n")
print("Tabela salva em tabela_melhor_k.tex")
