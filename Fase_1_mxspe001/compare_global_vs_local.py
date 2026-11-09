import pandas as pd

# Carrega os tempos uniformes e encontra o melhor por matriz
df_uniform = pd.read_csv("ilu_all_results.csv")
best_uniform = df_uniform.loc[df_uniform.groupby("matrix_id")["time"].idxmin()]
best_uniform = best_uniform[["matrix_id", "k", "time"]].rename(columns={"k": "k_uniform", "time": "time_uniform"})

# Carrega os tempos locais
df_local = pd.read_csv("local_times.csv")
df_local = df_local[["matrix_id", "k0", "k1", "time"]].rename(columns={"time": "time_local"})

# Une as tabelas
comparison = pd.merge(best_uniform, df_local, on="matrix_id")
comparison["gain"] = (comparison["time_uniform"] / comparison["time_local"]) - 1
comparison["gain_percent"] = comparison["gain"] * 100

# Ordena e exibe
comparison = comparison.sort_values("matrix_id")
print(comparison)

# Salva tabela LaTeX
with open("tabela_comparacao.tex", "w") as f:
    f.write("\\begin{table}[htbp]\n\\centering\n")
    f.write("\\caption{Comparação global vs local por matriz}\n")
    f.write("\\label{tab:comparacao_global_local}\n")
    f.write("\\begin{tabular}{c c c c c c}\n\\toprule\n")
    f.write("Matriz & k uniforme & Tempo uniforme (s) & k locais & Tempo local (s) & Ganho (\\%)\\\\\\midrule\n")
    for _, row in comparison.iterrows():
        f.write(f"{int(row['matrix_id'])} & {int(row['k_uniform'])} & {row['time_uniform']:.6f} & ({int(row['k0'])},{int(row['k1'])}) & {row['time_local']:.6f} & {row['gain_percent']:.2f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
    f.write("\\end{table}\n")

print("\nTabela LaTeX salva em tabela_comparacao.tex")
