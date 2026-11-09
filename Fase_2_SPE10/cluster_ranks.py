import os
import re
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Features de no que serao usadas no clustering
NODE_FEATURE_COLS = [
    "n_i", "nnz_ii", "nnz_if", "rho_i",
    "mean_degree", "std_degree", "r_if", "b_i",
    "w_if", "s_diag", "smalldiag_i",
    "min_delta", "p10_delta", "median_delta",
    "nnz_if_max", "cv_w",
]

# Features de aresta que serao agregadas
EDGE_FEATURE_COLS = [
    "log_nnz_ij", "log_w_ij", "r_nnz_ij", "r_w_ij",
]

def build_enriched_representation(df_nodes, df_edges):
    """
    Constroi a representacao enriquecida para cada no:
    x_til_i = [x_i, mean_{vizinhos} x_j, mean_{vizinhos} e_ij, max_{vizinhos} e_ij]
    """
    # Reindexar por rank 
    df_nodes = df_nodes.set_index("rank").sort_index()
    node_cols = [c for c in NODE_FEATURE_COLS if c in df_nodes.columns]
    x = df_nodes[node_cols].values
    n = len(df_nodes)
    node_ids = df_nodes.index.tolist()

    # Inicializar agregadores
    n_features = x.shape[1]
    mean_x_neigh = np.zeros((n, n_features))
    mean_e_ij = np.zeros((n, len(EDGE_FEATURE_COLS)))
    max_e_ij = np.zeros((n, len(EDGE_FEATURE_COLS)))
    n_neighbors = np.zeros(n)

    # Mapear rank -> indice
    rank_to_idx = {r: i for i, r in enumerate(node_ids)}

    # Para cada aresta, atualizar agregadores
    # Agrupando por src para calcular medias
    edge_cols = [c for c in EDGE_FEATURE_COLS if c in df_edges.columns]

    for src, group in df_edges.groupby("rank_src"):
        if src not in rank_to_idx:
            continue
        i = rank_to_idx[src]
        # Cada vizinho j
        for _, edge in group.iterrows():
            j = edge["rank_dst"]
            if j not in rank_to_idx:
                continue
            j_idx = rank_to_idx[j]
            # Agregar x_j
            mean_x_neigh[i] += x[j_idx]
            # Agregar e_ij
            edge_vals = edge[edge_cols].values.astype(float)
            mean_e_ij[i] += edge_vals
            max_e_ij[i] = np.maximum(max_e_ij[i], edge_vals)
            n_neighbors[i] += 1

    # Dividir pelas contagens para obter medias
    for i in range(n):
        if n_neighbors[i] > 0:
            mean_x_neigh[i] /= n_neighbors[i]
            mean_e_ij[i] /= n_neighbors[i]

    # Concatenar: [x_i, mean_x_neigh, mean_e_ij, max_e_ij]
    x_til = np.hstack([x, mean_x_neigh, mean_e_ij, max_e_ij])
    return x_til, node_ids


def cluster_one_matrix(df_nodes, df_edges, mat_id, p, verbose=False):
    """Clusteriza uma matriz para c em {round(sqrt(p))-1, round(sqrt(p)), round(sqrt(p))+1}."""
    x_til, node_ids = build_enriched_representation(df_nodes, df_edges)
    n_nodes = len(node_ids)

    if n_nodes < 3:
        return None

    # Padronizar
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_til)

    # Candidatos
    c_base = round(np.sqrt(p))
    c_candidates = sorted(set([max(2, c_base - 1), c_base, c_base + 1]))
    c_candidates = [c for c in c_candidates if c < n_nodes]

    if not c_candidates:
        return None

    best_score = -1
    best_c = None
    best_labels = None

    for c in c_candidates:
        try:
            km = KMeans(n_clusters=c, random_state=42, n_init=10)
            labels = km.fit_predict(x_scaled)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(x_scaled, labels)
            if verbose:
                print(f"  c={c}: silhouette={score:.4f}")
            if score > best_score:
                best_score = score
                best_c = c
                best_labels = labels
        except Exception as e:
            if verbose:
                print(f"  c={c}: erro {e}")

    if best_labels is None:
        return None

    return {
        "matrix_id": mat_id,
        "p": p,
        "n_nodes": n_nodes,
        "best_c": best_c,
        "silhouette": best_score,
        "labels": best_labels,
        "node_ids": node_ids,
    }


def main():
    base_dir = os.getcwd()
    pattern = "features_nodes_mat_*_p4.csv"
    node_files = sorted(glob.glob(pattern))

    if not node_files:
        print("Nenhum arquivo de features encontrado.")
        return

    results = []
    cluster_rows = []

    for node_file in node_files:
        match = re.search(r"features_nodes_mat_(\d+)_p(\d+)\.csv", node_file)
        if not match:
            continue
        mat_id = match.group(1)
        p = int(match.group(2))

        edge_file = f"features_edges_mat_{mat_id}_p{p}.csv"
        if not os.path.exists(edge_file):
            print(f"AVISO: {edge_file} nao encontrado, pulando {mat_id}")
            continue

        df_nodes = pd.read_csv(node_file)
        df_edges = pd.read_csv(edge_file)

        print(f"Clustering system_{mat_id} (p={p}, {len(df_nodes)} nos, {len(df_edges)} arestas)")

        res = cluster_one_matrix(df_nodes, df_edges, mat_id, p, verbose=True)
        if res is None:
            print(f"  Falhou para system_{mat_id}")
            continue

        print(f"  Melhor c={res['best_c']} (silhouette={res['silhouette']:.4f})")
        print(f"  Distribuicao dos clusters:")
        unique, counts = np.unique(res["labels"], return_counts=True)
        for u, cnt in zip(unique, counts):
            print(f"    Cluster {u}: {cnt} nos")

        # Salvar atribuicoes
        for node_id, label in zip(res["node_ids"], res["labels"]):
            cluster_rows.append({
                "matrix_id": mat_id,
                "rank": node_id,
                "cluster": int(label),
            })

        results.append({
            "matrix_id": mat_id,
            "p": p,
            "n_nodes": res["n_nodes"],
            "best_c": res["best_c"],
            "silhouette": res["silhouette"],
        })
        print()

    # Salvar resultados
    df_res = pd.DataFrame(results)
    df_res.to_csv("cluster_summary.csv", index=False)
    print("Resumo salvo em cluster_summary.csv")
    print(df_res.to_string(index=False))

    df_clusters = pd.DataFrame(cluster_rows)
    df_clusters.to_csv("cluster_assignments.csv", index=False)
    print("\nAtribuicoes salvas em cluster_assignments.csv")
    print(f"Total de nos clusterizados: {len(df_clusters)}")


if __name__ == "__main__":
    main()
