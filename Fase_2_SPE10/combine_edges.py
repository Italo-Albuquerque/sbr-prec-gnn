import os
import pandas as pd

p = 4
edge_files = sorted([f for f in os.listdir(".") 
                     if f.startswith("features_edges_mat_") and f.endswith(f"_p{p}.csv")])

if not edge_files:
    print(f"Nenhum arquivo features_edges_mat_*_p{p}.csv encontrado")
    exit(1)

dfs = []
for f in edge_files:
    mat_id = int(f.split("_")[3])
    df = pd.read_csv(f)
    df["matrix_id"] = mat_id
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all.to_csv(f"features_edges_all_p{p}.csv", index=False)
print(f"Salvo features_edges_all_p{p}.csv com {len(df_all)} arestas de {len(edge_files)} matrizes")
