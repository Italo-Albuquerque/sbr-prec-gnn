import pandas as pd

df = pd.read_csv("resultado_k012_conjuntoA.csv")
df_ok = df[df["status"] == "OK"]
grouped = df_ok.groupby(["matrix_id", "k"])["time"].mean().reset_index()

print("Matriz & $k=0$ (s) & $k=1$ (s) & $k=2$ (s) \\\\ \\midrule")
for mat in sorted(df["matrix_id"].unique()):
    t0 = grouped[(grouped["matrix_id"] == mat) & (grouped["k"] == 0)]["time"].values[0]
    t1 = grouped[(grouped["matrix_id"] == mat) & (grouped["k"] == 1)]["time"].values[0]
    t2 = grouped[(grouped["matrix_id"] == mat) & (grouped["k"] == 2)]["time"].values[0]
    
    m = min(t0, t1, t2)
    s0 = f"\\textbf{{{t0:.3f}}}".replace('.', ',') if t0 == m else f"{t0:.3f}".replace('.', ',')
    s1 = f"\\textbf{{{t1:.3f}}}".replace('.', ',') if t1 == m else f"{t1:.3f}".replace('.', ',')
    s2 = f"\\textbf{{{t2:.3f}}}".replace('.', ',') if t2 == m else f"{t2:.3f}".replace('.', ',')
    
    print(f"{mat} & {s0} & {s1} & {s2} \\\\")
