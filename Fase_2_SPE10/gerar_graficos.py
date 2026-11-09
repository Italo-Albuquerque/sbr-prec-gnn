import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUT_DIR = "figuras_spe10"
os.makedirs(OUT_DIR, exist_ok=True)

BASELINE_CSV = "resultado_k012_conjuntoA.csv"
CLUSTER_SUMMARY = "cluster_summary.csv"
TEACHER_LABELS = "teacher_soft_labels.csv"
TEACHER_CONFIGS = "teacher_all_configs.csv"
TRAIN_RESULTS = "training_results.csv"
GNN_SAGE = "gnn_results_sage_p4.csv"
GNN_GINE = "gnn_results_gine_p4.csv"


def plot_baseline():
    if not os.path.exists(BASELINE_CSV):
        print(f"  [SKIP] {BASELINE_CSV} não encontrado")
        return

    df = pd.read_csv(BASELINE_CSV)
    df_ok = df[df["status"] == "OK"].copy()
    # Tira a média dos 3 RHS primeiro
    grouped = df_ok.groupby(["matrix_id", "k"])["time"].mean().reset_index()

    matrizes = sorted(df["matrix_id"].unique())
    ks = [0, 1, 2]
    cores = {0: "#d62728", 1: "#1f77b4", 2: "#2ca02c"}
    labels = {0: "k=0", 1: "k=1", 2: "k=2"}

    fig, ax = plt.subplots(figsize=(max(12, len(matrizes) * 0.9), 5))
    x = np.arange(len(matrizes))
    width = 0.25

    max_time = grouped["time"].max() if len(grouped) > 0 else 100
    y_x = max_time * 0.02

    for i, k in enumerate(ks):
        tempos = []
        for mat in matrizes:
            sub = grouped[(grouped["matrix_id"] == mat) & (grouped["k"] == k)]
            if len(sub) > 0:
                tempos.append(sub["time"].values[0])
            else:
                tempos.append(0)
        ax.bar(x + i * width, tempos, width, label=labels[k],
               color=cores[k], edgecolor="black", linewidth=0.5)

        for j, mat in enumerate(matrizes):
            sub = df[(df["matrix_id"] == mat) & (df["k"] == k)]
            n_falhas = (sub["status"] == "FALHOU").sum()
            if n_falhas > 0:
                ax.text(x[j] + i * width, y_x, "X", ha="center", va="bottom",
                        fontsize=9, color="red", fontweight="bold")

    ax.set_xlabel("Matriz")
    ax.set_ylabel("Tempo médio em segundos")
    ax.set_title("Baseline Global: tempo médio de resolução por nível de preenchimento k")
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(m) for m in matrizes], rotation=45, ha="right")
    ax.legend(title="Nível de preenchimento")
    ax.grid(axis="y", alpha=0.3)

    plt.savefig(os.path.join(OUT_DIR, "baseline_tempo_por_k.png"))
    plt.close()
    print("  [OK] baseline_tempo_por_k.png")


def plot_silhouette():
    if not os.path.exists(CLUSTER_SUMMARY):
        print(f"  [SKIP] {CLUSTER_SUMMARY} não encontrado")
        return

    df = pd.read_csv(CLUSTER_SUMMARY)

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.7), 4))
    x = np.arange(len(df))
    ax.bar(x, df["silhouette"], color="#4c72b0", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Matriz")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Qualidade da Clusterização por Matriz\nTodas as matrizes resultaram em 2 clusters")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(m)) for m in df["matrix_id"]], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.savefig(os.path.join(OUT_DIR, "silhouette_por_matriz.png"))
    plt.close()
    print("  [OK] silhouette_por_matriz.png")


def plot_teacher_pi():
    if not os.path.exists(TEACHER_LABELS):
        print(f"  [SKIP] {TEACHER_LABELS} não encontrado")
        return

    df = pd.read_csv(TEACHER_LABELS)
    df["label"] = df["matrix_id"].astype(str) + " c" + df["cluster"].astype(str)

    fig, ax = plt.subplots(figsize=(max(14, len(df) * 0.5), 5))
    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df["pi_k0"], width, label="k=0", color="#d62728",
           edgecolor="black", linewidth=0.5)
    ax.bar(x, df["pi_k1"], width, label="k=1", color="#1f77b4",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width, df["pi_k2"], width, label="k=2", color="#2ca02c",
           edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Probabilidade $\\pi_k$")
    ax.set_title("Distribuição de probabilidades $\\pi_k$ por cluster do modelo Teacher")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=8)
    ax.legend(title="Nível de preenchimento")
    ax.grid(axis="y", alpha=0.3)

    plt.savefig(os.path.join(OUT_DIR, "teacher_pi_por_cluster.png"))
    plt.close()
    print("  [OK] teacher_pi_por_cluster.png")


def plot_mae_comparativo():
    dados = []

    if os.path.exists(TRAIN_RESULTS):
        df = pd.read_csv(TRAIN_RESULTS)
        dados.append(("Random Forest", df["mae_rf_q1"].mean(), df["mae_rf_q2"].mean()))
        dados.append(("MLP", df["mae_mlp_q1"].mean(), df["mae_mlp_q2"].mean()))

    if os.path.exists(GNN_SAGE):
        df = pd.read_csv(GNN_SAGE)
        dados.append(("GraphSAGE", df["mae_q1"].mean(), df["mae_q2"].mean()))

    if os.path.exists(GNN_GINE):
        df = pd.read_csv(GNN_GINE)
        dados.append(("GINE", df["mae_q1"].mean(), df["mae_q2"].mean()))

    if not dados:
        print("  [SKIP] Nenhum arquivo de resultados de modelos encontrado")
        return

    modelos = [d[0] for d in dados]
    mae_q1 = [d[1] for d in dados]
    mae_q2 = [d[2] for d in dados]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(modelos))
    width = 0.35

    bars1 = ax.bar(x - width / 2, mae_q1, width, label="$q_1$ = probabilidade de k maior ou igual a 1",
                   color="#4c72b0", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, mae_q2, width, label="$q_2$ = probabilidade de k igual a 2",
                   color="#dd8452", edgecolor="black", linewidth=0.5)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ymax = max(max(mae_q1), max(mae_q2)) * 1.35
    ax.set_ylim(0, ymax)

    ax.set_xlabel("Modelo")
    ax.set_ylabel("MAE médio com validação leave-one-case-out")
    ax.set_title("Comparativo de MAE entre modelos preditivos com p = 4")
    ax.set_xticks(x)
    ax.set_xticklabels(modelos)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    plt.savefig(os.path.join(OUT_DIR, "mae_comparativo_modelos.png"))
    plt.close()
    print("  [OK] mae_comparativo_modelos.png")


def plot_ganho_adaptativo():
    if not os.path.exists(TEACHER_CONFIGS) or not os.path.exists(BASELINE_CSV):
        print("  [SKIP] Arquivos não encontrados")
        return

    df = pd.read_csv(TEACHER_CONFIGS)
    df_ok = df[df["converged"] == True]
    best_local = df_ok.groupby("matrix_id")["time"].min()

    baseline = pd.read_csv(BASELINE_CSV)
    baseline_ok = baseline[baseline["status"] == "OK"]
    #Calcula a média dos 3 RHS para cada k, e depois o menor tempo entre os k
    mean_baseline = baseline_ok.groupby(["matrix_id", "k"])["time"].mean().reset_index()
    best_uniform = mean_baseline.groupby("matrix_id")["time"].min()

    common = sorted(best_local.index.intersection(best_uniform.index))

    if len(common) == 0:
        print("  [SKIP] Nenhuma matriz em comum entre teacher e baseline")
        return

    fig, ax = plt.subplots(figsize=(max(11, len(common) * 0.8), 4.5))
    x = np.arange(len(common))
    width = 0.35

    ax.bar(x - width / 2, best_uniform[common].values, width,
           label="Melhor configuração uniforme", color="#8c8c8c", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, best_local[common].values, width,
           label="Melhor configuração adaptativa", color="#2ca02c", edgecolor="black", linewidth=0.5)

    ymax = max(best_uniform[common].max(), best_local[common].max()) * 1.40
    ax.set_ylim(0, ymax)

    ax.set_xlabel("Matriz")
    ax.set_ylabel("Tempo em segundos")
    ax.set_title("Comparativo entre a melhor configuração uniforme e a adaptativa")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(m)) for m in common], rotation=45, ha="right")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    plt.savefig(os.path.join(OUT_DIR, "ganho_adaptativo.png"))
    plt.close()
    print("  [OK] ganho_adaptativo.png")

def main():
    print("Gerando figuras para a dissertação do Conjunto A...")
    print(f"Saída: {OUT_DIR}/\n")

    plot_baseline()
    plot_silhouette()
    plot_teacher_pi()
    plot_mae_comparativo()
    plot_ganho_adaptativo()

    print(f"\nConcluído! Figuras salvas em {OUT_DIR}/")

if __name__ == "__main__":
    main()
