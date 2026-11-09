import pandas as pd

df = pd.read_csv("ilu_all_results.csv")

dataset = []

for mat in df["matrix_id"].unique():
    sub = df[df["matrix_id"] == mat]
    n = int(sub["n"].iloc[0])
    nnz = int(sub["nnz"].iloc[0])
    density = nnz / (n * n)
    
    # Muda de its para time
    mean_time = sub.groupby("k")["time"].mean().reset_index()
    best_k = int(mean_time.loc[mean_time["time"].idxmin(), "k"])
    
    dataset.append({
        "matrix_id": mat,
        "n": n,
        "nnz": nnz,
        "density": density,
        "best_k": best_k
    })

dataset = pd.DataFrame(dataset)
dataset.to_csv("dataset.csv", index=False)
print(dataset)
