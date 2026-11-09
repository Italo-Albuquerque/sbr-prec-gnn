import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

NODE_FEATURE_COLS = [
    "n_i", "nnz_ii", "nnz_if", "rho_i",
    "mean_degree", "std_degree", "r_if", "b_i",
    "w_if", "s_diag", "smalldiag_i",
    "min_delta", "p10_delta", "median_delta",
    "nnz_if_max", "cv_w",
]

EDGE_FEATURE_COLS = [
    "log_nnz_ij", "log_w_ij", "r_nnz_ij", "r_w_ij",
]

def build_cluster_descriptor(df_nodes, df_edges, cluster_assignments):
    descriptors = []
    for (mat_id, cluster_id), _ in cluster_assignments.groupby(["matrix_id", "cluster"]):
        node_ranks = cluster_assignments[
            (cluster_assignments["matrix_id"] == mat_id) &
            (cluster_assignments["cluster"] == cluster_id)
        ]["rank"].values

        df_nodes_sub = df_nodes[df_nodes["rank"].isin(node_ranks)]
        df_edges_int = df_edges[
            (df_edges["rank_src"].isin(node_ranks)) &
            (df_edges["rank_dst"].isin(node_ranks))
        ]
        df_edges_cut = df_edges[
            (df_edges["rank_src"].isin(node_ranks)) &
            (~df_edges["rank_dst"].isin(node_ranks))
        ]

        feat = {}
        n_nodes_total = cluster_assignments[cluster_assignments["matrix_id"] == mat_id]["rank"].nunique()
        feat["cluster_size_ratio"] = len(node_ranks) / max(1, n_nodes_total)

        for col in NODE_FEATURE_COLS:
            if col in df_nodes_sub.columns and len(df_nodes_sub) > 0:
                vals = df_nodes_sub[col].values
                feat[f"node_{col}_mean"] = np.mean(vals)
                feat[f"node_{col}_std"] = np.std(vals)
                feat[f"node_{col}_min"] = np.min(vals)
                feat[f"node_{col}_max"] = np.max(vals)
            else:
                feat[f"node_{col}_mean"] = 0.0
                feat[f"node_{col}_std"] = 0.0
                feat[f"node_{col}_min"] = 0.0
                feat[f"node_{col}_max"] = 0.0

        for col in EDGE_FEATURE_COLS:
            if col in df_edges_int.columns and len(df_edges_int) > 0:
                vals = df_edges_int[col].values
                feat[f"edge_int_{col}_mean"] = np.mean(vals)
                feat[f"edge_int_{col}_max"] = np.max(vals)
            else:
                feat[f"edge_int_{col}_mean"] = 0.0
                feat[f"edge_int_{col}_max"] = 0.0

        for col in EDGE_FEATURE_COLS:
            if col in df_edges_cut.columns and len(df_edges_cut) > 0:
                vals = df_edges_cut[col].values
                feat[f"edge_cut_{col}_mean"] = np.mean(vals)
                feat[f"edge_cut_{col}_max"] = np.max(vals)
            else:
                feat[f"edge_cut_{col}_mean"] = 0.0
                feat[f"edge_cut_{col}_max"] = 0.0

        feat["matrix_id"] = mat_id
        feat["cluster"] = cluster_id
        descriptors.append(feat)

    return pd.DataFrame(descriptors)

def main():
    print("Carregando dados...")
    df_edges = pd.read_csv("features_edges_all_p4.csv")

    df_nodes_list = []
    for f in sorted(os.listdir(".")):
        if f.startswith("features_nodes_mat_") and f.endswith("_p4.csv"):
            df = pd.read_csv(f)
            mat_id = int(f.split("_")[3])
            df["matrix_id"] = mat_id
            df_nodes_list.append(df)

    df_nodes = pd.concat(df_nodes_list, ignore_index=True)
    df_clusters = pd.read_csv("cluster_assignments.csv")
    df_labels = pd.read_csv("teacher_soft_labels.csv")

    print("\nConstruindo descritores de cluster...")
    df_desc = build_cluster_descriptor(df_nodes, df_edges, df_clusters)
    df_data = df_desc.merge(df_labels, on=["matrix_id", "cluster"])

    feature_cols = [c for c in df_data.columns if c not in ["matrix_id", "cluster", "pi_k0", "pi_k1", "pi_k2", "q_g1", "q_g2"]]
    X = df_data[feature_cols].values
    y_q1 = df_data["q_g1"].values
    y_q2 = df_data["q_g2"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calcular preditor trivial (média) para referência
    mae_trivial_q1 = mean_absolute_error(y_q1, [y_q1.mean()] * len(y_q1))
    mae_trivial_q2 = mean_absolute_error(y_q2, [y_q2.mean()] * len(y_q2))
    print(f"\n=== Preditor Trivial (Média) ===")
    print(f"MAE Trivial q1: {mae_trivial_q1:.4f}")
    print(f"MAE Trivial q2: {mae_trivial_q2:.4f}\n")

    mat_ids = df_data["matrix_id"].unique()
    print(f"Validacao leave-one-case-out com {len(mat_ids)} matrizes")

    results = []

    for test_mat in mat_ids:
        train_mask = df_data["matrix_id"] != test_mat
        test_mask = df_data["matrix_id"] == test_mat

        X_train = X_scaled[train_mask]
        X_test = X_scaled[test_mask]
        q1_train = y_q1[train_mask]
        q1_test = y_q1[test_mask]
        q2_train = y_q2[train_mask]
        q2_test = y_q2[test_mask]

        if len(X_train) < 5 or len(X_test) < 1:
            continue

        # Random Forest Regressor
        rf1 = RandomForestRegressor(n_estimators=100, random_state=42)
        rf2 = RandomForestRegressor(n_estimators=100, random_state=42)

        rf1.fit(X_train, q1_train)
        pred_rf1 = rf1.predict(X_test)
        rf2.fit(X_train, q2_train)
        pred_rf2 = rf2.predict(X_test)

        mae_rf1 = mean_absolute_error(q1_test, pred_rf1)
        mae_rf2 = mean_absolute_error(q2_test, pred_rf2)

        # MLP
        mlp1 = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
        mlp2 = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)

        try:
            mlp1.fit(X_train, q1_train)
            pred_mlp1 = mlp1.predict(X_test)
            mlp2.fit(X_train, q2_train)
            pred_mlp2 = mlp2.predict(X_test)

            mae_mlp1 = mean_absolute_error(q1_test, pred_mlp1)
            mae_mlp2 = mean_absolute_error(q2_test, pred_mlp2)
        except Exception:
            mae_mlp1 = float("nan")
            mae_mlp2 = float("nan")

        results.append({
            "test_matrix": test_mat,
            "mae_rf_q1": mae_rf1,
            "mae_rf_q2": mae_rf2,
            "mae_mlp_q1": mae_mlp1,
            "mae_mlp_q2": mae_mlp2,
        })

        print(f"  Test {test_mat}: RF(q1={mae_rf1:.3f}, q2={mae_rf2:.3f}) MLP(q1={mae_mlp1:.3f}, q2={mae_mlp2:.3f})")

    df_results = pd.DataFrame(results)
    print("\n=== Resumo Final ===")
    print(f"MAE medio RF:  q1={df_results['mae_rf_q1'].mean():.4f}, q2={df_results['mae_rf_q2'].mean():.4f}")
    print(f"MAE medio MLP: q1={df_results['mae_mlp_q1'].mean():.4f}, q2={df_results['mae_mlp_q2'].mean():.4f}")

    df_results.to_csv("training_results.csv", index=False)
    print("\nResultados salvos em training_results.csv")

if __name__ == "__main__":
    main()
