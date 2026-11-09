import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GINEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Configuracao
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

HIDDEN_DIM = 32
EPOCHS = 200
LR = 0.01
SEED = 42


def load_data(p):
    """Carrega features, clusters e rotulos para um dado p."""
    # Features dos nos
    node_files = sorted([f for f in os.listdir(".") if f.startswith("features_nodes_mat_") and f.endswith(f"_p{p}.csv")])
    edge_files = sorted([f for f in os.listdir(".") if f.startswith("features_edges_mat_") and f.endswith(f"_p{p}.csv")])

    if not node_files or not edge_files:
        print(f"Nenhum arquivo de features para p={p}")
        sys.exit(1)

    df_nodes_list = []
    for f in node_files:
        mat_id = int(f.split("_")[3])
        df = pd.read_csv(f)
        df["matrix_id"] = mat_id
        df_nodes_list.append(df)
    df_nodes = pd.concat(df_nodes_list, ignore_index=True)

    df_edges_list = []
    for f in edge_files:
        mat_id = int(f.split("_")[3])
        df = pd.read_csv(f)
        df["matrix_id"] = mat_id
        df_edges_list.append(df)
    df_edges = pd.concat(df_edges_list, ignore_index=True)

    df_clusters = pd.read_csv("cluster_assignments.csv")
    df_labels = pd.read_csv("teacher_soft_labels.csv")

    return df_nodes, df_edges, df_clusters, df_labels


def build_graphs(df_nodes, df_edges, df_clusters, df_labels):
    """Constroi um grafo por matriz."""
    graphs = []

    for mat_id in sorted(df_nodes["matrix_id"].unique()):
        # Nos dessa matriz
        df_n = df_nodes[df_nodes["matrix_id"] == mat_id].sort_values("rank").reset_index(drop=True)
        # Arestas dessa matriz
        df_e = df_edges[df_edges["matrix_id"] == mat_id].reset_index(drop=True)
        # Clusters dessa matriz
        df_c = df_clusters[df_clusters["matrix_id"] == mat_id].sort_values("rank").reset_index(drop=True)
        # Rotulos dessa matriz
        df_l = df_labels[df_labels["matrix_id"] == mat_id].sort_values("cluster").reset_index(drop=True)

        # Verificar consistencia
        if len(df_n) != len(df_c):
            print(f"  AVISO: {mat_id} tem {len(df_n)} nos e {len(df_c)} clusters")
            continue

        # Features dos nos
        x = df_n[NODE_FEATURE_COLS].values.astype(np.float32)

        # Cluster id de cada no
        cluster_ids = df_c["cluster"].values.astype(np.int64)

        # Arestas
        if len(df_e) > 0:
            # Mapear rank -> indice no grafo
            rank_to_idx = {r: i for i, r in enumerate(df_n["rank"].values)}
            edge_index_list = []
            edge_attr_list = []
            for _, edge in df_e.iterrows():
                src = edge["rank_src"]
                dst = edge["rank_dst"]
                if src in rank_to_idx and dst in rank_to_idx:
                    edge_index_list.append([rank_to_idx[src], rank_to_idx[dst]])
                    edge_attr_list.append(edge[EDGE_FEATURE_COLS].values.astype(np.float32))

            if edge_index_list:
                edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(np.array(edge_attr_list), dtype=torch.float)
                # Adicionar arestas reversas para propagacao bidirecional
                rev_edge_index = edge_index.flip(0)
                edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, len(EDGE_FEATURE_COLS)), dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(EDGE_FEATURE_COLS)), dtype=torch.float)

        # Rotulos: q_g1 e q_g2 para cada cluster
        n_clusters = int(cluster_ids.max()) + 1
        y_q1 = np.zeros(n_clusters, dtype=np.float32)
        y_q2 = np.zeros(n_clusters, dtype=np.float32)
        for _, row in df_l.iterrows():
            c = int(row["cluster"])
            if c < n_clusters:
                y_q1[c] = row["q_g1"]
                y_q2[c] = row["q_g2"]

        graph = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            cluster_ids=torch.tensor(cluster_ids, dtype=torch.long),
            y=torch.tensor(np.stack([y_q1, y_q2], axis=1), dtype=torch.float),
            matrix_id=mat_id,
        )
        graphs.append(graph)

    return graphs


class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden, edge_dim=None):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, edge_index, cluster_ids):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Pooling por cluster
        n_clusters = int(cluster_ids.max().item()) + 1
        cluster_states = []
        for c in range(n_clusters):
            mask = cluster_ids == c
            if mask.sum() > 0:
                cluster_states.append(x[mask].mean(dim=0))
            else:
                cluster_states.append(torch.zeros(x.shape[1], device=x.device))
        h_g = torch.stack(cluster_states)

        out = torch.sigmoid(self.head(h_g))
        return out


class GINEModel(nn.Module):
    def __init__(self, in_channels, hidden, edge_dim):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_channels, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        nn2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.conv1 = GINEConv(nn=nn1, edge_dim=edge_dim)
        self.conv2 = GINEConv(nn=nn2, edge_dim=edge_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, edge_index, edge_attr, cluster_ids):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        n_clusters = int(cluster_ids.max().item()) + 1
        cluster_states = []
        for c in range(n_clusters):
            mask = cluster_ids == c
            if mask.sum() > 0:
                cluster_states.append(x[mask].mean(dim=0))
            else:
                cluster_states.append(torch.zeros(x.shape[1], device=x.device))
        h_g = torch.stack(cluster_states)

        out = torch.sigmoid(self.head(h_g))
        return out


def standardize_graphs(graphs, node_scaler=None, edge_scaler=None, fit=True):
    """Padroniza features dos nos e arestas."""
    if fit:
        all_x = torch.cat([g.x for g in graphs], dim=0).numpy()
        node_scaler = StandardScaler().fit(all_x)
        all_e = []
        for g in graphs:
            if g.edge_attr.numel() > 0:
                all_e.append(g.edge_attr.numpy())
        if all_e:
            all_e = np.concatenate(all_e, axis=0)
            edge_scaler = StandardScaler().fit(all_e)
        else:
            edge_scaler = None

    for g in graphs:
        g.x = torch.tensor(node_scaler.transform(g.x.numpy()), dtype=torch.float)
        if g.edge_attr.numel() > 0 and edge_scaler is not None:
            g.edge_attr = torch.tensor(edge_scaler.transform(g.edge_attr.numpy()), dtype=torch.float)

    return graphs, node_scaler, edge_scaler


def train_one_model(model, graphs_train, graphs_test, model_type, epochs=EPOCHS, lr=LR):
    """Treina um modelo e retorna as predicoes no grafo de teste."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for g in graphs_train:
            optimizer.zero_grad()
            if model_type == "sage":
                out = model(g.x, g.edge_index, g.cluster_ids)
            else:
                out = model(g.x, g.edge_index, g.edge_attr, g.cluster_ids)
            loss = criterion(out, g.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Predicao
    model.eval()
    predictions = []
    with torch.no_grad():
        for g in graphs_test:
            if model_type == "sage":
                out = model(g.x, g.edge_index, g.cluster_ids)
            else:
                out = model(g.x, g.edge_index, g.edge_attr, g.cluster_ids)
            predictions.append((g.matrix_id, out.numpy(), g.y.numpy()))

    return predictions


def main():
    if len(sys.argv) > 1:
        p = int(sys.argv[1])
    else:
        p = 4

    print(f"Treinando GNNs com p={p}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Carregar dados
    df_nodes, df_edges, df_clusters, df_labels = load_data(p)
    print(f"  Nos: {len(df_nodes)}")
    print(f"  Arestas: {len(df_edges)}")
    print(f"  Clusters: {len(df_clusters)}")
    print(f"  Rotulos: {len(df_labels)}")

    # Construir grafos
    graphs = build_graphs(df_nodes, df_edges, df_clusters, df_labels)
    print(f"  Grafos: {len(graphs)}")

    if len(graphs) < 3:
        print("Poucos grafos para validacao leave-one-case-out")
        return

    # Leave-one-case-out
    results_sage = []
    results_gine = []

    mat_ids = [g.matrix_id for g in graphs]
    print(f"\nValidacao leave-one-case-out com {len(graphs)} grafos")

    for i, test_mat in enumerate(mat_ids):
        graphs_train = [g for j, g in enumerate(graphs) if j != i]
        graphs_test = [graphs[i]]

        # Padronizar com base no treino
        graphs_train_std, node_scaler, edge_scaler = standardize_graphs(graphs_train, fit=True)
        graphs_test_std = [
            Data(
                x=torch.tensor(node_scaler.transform(g.x.numpy()), dtype=torch.float),
                edge_index=g.edge_index,
                edge_attr=torch.tensor(edge_scaler.transform(g.edge_attr.numpy()), dtype=torch.float) if g.edge_attr.numel() > 0 and edge_scaler is not None else g.edge_attr,
                cluster_ids=g.cluster_ids,
                y=g.y,
                matrix_id=g.matrix_id,
            )
            for g in graphs_test
        ]

        in_channels = graphs_train_std[0].x.shape[1]
        edge_dim = graphs_train_std[0].edge_attr.shape[1] if graphs_train_std[0].edge_attr.numel() > 0 else len(EDGE_FEATURE_COLS)

        # GraphSAGE
        model_sage = GraphSAGEModel(in_channels, HIDDEN_DIM)
        pred_sage = train_one_model(model_sage, graphs_train_std, graphs_test_std, "sage")

        # GINE
        model_gine = GINEModel(in_channels, HIDDEN_DIM, edge_dim)
        pred_gine = train_one_model(model_gine, graphs_train_std, graphs_test_std, "gine")

        for mat_id, pred, true in pred_sage:
            mae_q1 = mean_absolute_error(true[:, 0], pred[:, 0])
            mae_q2 = mean_absolute_error(true[:, 1], pred[:, 1])
            results_sage.append({
                "test_matrix": mat_id,
                "mae_q1": mae_q1,
                "mae_q2": mae_q2,
            })

        for mat_id, pred, true in pred_gine:
            mae_q1 = mean_absolute_error(true[:, 0], pred[:, 0])
            mae_q2 = mean_absolute_error(true[:, 1], pred[:, 1])
            results_gine.append({
                "test_matrix": mat_id,
                "mae_q1": mae_q1,
                "mae_q2": mae_q2,
            })

        print(f"  Test {test_mat}: SAGE(q1={results_sage[-1]['mae_q1']:.3f}, q2={results_sage[-1]['mae_q2']:.3f}) "
              f"GINE(q1={results_gine[-1]['mae_q1']:.3f}, q2={results_gine[-1]['mae_q2']:.3f})")

    # Resumo
    df_sage = pd.DataFrame(results_sage)
    df_gine = pd.DataFrame(results_gine)

    print("\n=== Resumo GraphSAGE ===")
    print(f"MAE medio q1: {df_sage['mae_q1'].mean():.4f}")
    print(f"MAE medio q2: {df_sage['mae_q2'].mean():.4f}")

    print("\n=== Resumo GINE ===")
    print(f"MAE medio q1: {df_gine['mae_q1'].mean():.4f}")
    print(f"MAE medio q2: {df_gine['mae_q2'].mean():.4f}")

    # Salvar
    df_sage.to_csv(f"gnn_results_sage_p{p}.csv", index=False)
    df_gine.to_csv(f"gnn_results_gine_p{p}.csv", index=False)
    print(f"\nResultados salvos em gnn_results_sage_p{p}.csv e gnn_results_gine_p{p}.csv")


if __name__ == "__main__":
    main()
