import pandas as pd

baseline = pd.read_csv("resultado_k012_conjuntoA.csv")
baseline_ok = baseline[baseline["status"] == "OK"]
best_uniform = baseline_ok.groupby("matrix_id")["time"].min()

teacher = pd.read_csv("teacher_all_configs.csv")
teacher_ok = teacher[teacher["converged"] == True]
best_local = teacher_ok.groupby("matrix_id")["time"].min()

matrizes = sorted(best_uniform.index)
print("Matriz & Melhor uniforme (s) & Melhor local (s) & Ganho (\\%) \\\\ \\midrule")
ganhos = []
for m in matrizes:
    u = best_uniform[m]
    l = best_local[m]
    g = (u / l - 1) * 100
    ganhos.append(g)
    print(f"{m} & {u:.3f} & {l:.3f} & {g:+.2f} \\\\")

print(f"\\midrule\nMédia & --- & --- & {sum(ganhos)/len(ganhos):+.2f} \\\\")
