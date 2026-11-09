import pandas as pd

# Lê o resultado do Teacher
df = pd.read_csv("teacher_all_configs.csv")
df_ok = df[df["converged"] == True]
best_local = df_ok.groupby("matrix_id")["time"].min()

# Lê o baseline para comparar com o k=2
baseline = pd.read_csv("resultado_k012_conjuntoA.csv")
k2_baseline = baseline[(baseline["k"] == 2) & (baseline["status"] == "OK")].groupby("matrix_id")["time"].mean()

print("Matriz | Melhor Local (Teacher) | Tempo K=2 (Baseline) | São iguais?")
for m in sorted(best_local.index):
    l = best_local[m]
    k2 = k2_baseline[m]
    iguais = "SIM (ERRO!)" if abs(l - k2) < 0.001 else "Não"
    print(f"{m:>3} | {l:>8.3f} | {k2:>8.3f} | {iguais}")
