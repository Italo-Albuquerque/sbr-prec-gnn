import sys
import os
import gc
import numpy as np
import pandas as pd
from petsc4py import PETSc
from mpi4py import MPI


def compute_node_features(A, comm, n, rank, size):
    rstart, rend = A.getOwnershipRange()
    n_local = rend - rstart

    # Obter CSR local: ia (indptr), ja (indices), aa (valores)
    ia, ja, aa = A.getValuesCSR()

    nnz_ii = 0
    nnz_if = 0
    sum_abs_ii = 0.0
    sum_abs_if = 0.0
    diag_sum = 0.0
    row_degrees = []
    diag_dominance = []
    boundary_rows = 0
    small_diag_count = 0

    neighbors = {}

    for local_row in range(n_local):
        start = ia[local_row]
        end = ia[local_row + 1]
        global_row = rstart + local_row

        row_degree = 0
        row_diag = 0.0
        row_off_diag_sum = 0.0
        has_interface = False

        for k in range(start, end):
            col = int(ja[k])
            val = float(aa[k])
            owner = int(col * size // n)
            abs_val = abs(val)
            row_degree += 1

            if owner == rank:
                nnz_ii += 1
                sum_abs_ii += abs_val
                if col == global_row:
                    row_diag = abs_val
                    diag_sum += abs_val
                else:
                    row_off_diag_sum += abs_val
            else:
                nnz_if += 1
                sum_abs_if += abs_val
                has_interface = True
                if owner not in neighbors:
                    neighbors[owner] = {"nnz": 0, "w": 0.0, "max_abs": 0.0}
                neighbors[owner]["nnz"] += 1
                neighbors[owner]["w"] += abs_val
                if abs_val > neighbors[owner]["max_abs"]:
                    neighbors[owner]["max_abs"] = abs_val

        row_degrees.append(row_degree)

        if row_diag > 0:
            delta = row_diag / (row_off_diag_sum + 1e-12)
            diag_dominance.append(delta)
            if delta < 0.1:
                small_diag_count += 1
        else:
            diag_dominance.append(0.0)

        if has_interface:
            boundary_rows += 1

    mean_degree = float(np.mean(row_degrees)) if row_degrees else 0.0
    std_degree = float(np.std(row_degrees)) if row_degrees else 0.0

    rho_i = nnz_ii / (n_local * n_local + 1e-12)
    r_if = nnz_if / (nnz_ii + 1e-12)
    b_i = boundary_rows / (n_local + 1e-12)
    w_if = sum_abs_if / (sum_abs_ii + 1e-12)
    s_diag = diag_sum / (sum_abs_ii + 1e-12)
    smalldiag_i = small_diag_count / (n_local + 1e-12)

    diag_array = np.array(diag_dominance)
    min_delta = float(np.min(diag_array)) if len(diag_array) > 0 else 0.0
    p10_delta = float(np.percentile(diag_array, 10)) if len(diag_array) > 0 else 0.0
    median_delta = float(np.median(diag_array)) if len(diag_array) > 0 else 0.0

    deg_i = len(neighbors)
    nnz_if_max = max([v["nnz"] for v in neighbors.values()], default=0)

    weights = [v["w"] for v in neighbors.values()]
    if len(weights) > 1:
        cv_w = float(np.std(weights) / (np.mean(weights) + 1e-12))
    else:
        cv_w = 0.0

    node_feat = {
        "rank": rank,
        "n_i": n_local,
        "nnz_ii": nnz_ii,
        "nnz_if": nnz_if,
        "rho_i": rho_i,
        "mean_degree": mean_degree,
        "std_degree": std_degree,
        "r_if": r_if,
        "b_i": b_i,
        "deg_i": deg_i,
        "w_if": w_if,
        "s_diag": s_diag,
        "smalldiag_i": smalldiag_i,
        "min_delta": min_delta,
        "p10_delta": p10_delta,
        "median_delta": median_delta,
        "nnz_if_max": nnz_if_max,
        "cv_w": cv_w,
    }

    edge_feats = []
    for j, info in neighbors.items():
        edge_feats.append({
            "rank_src": rank,
            "rank_dst": j,
            "nnz_ij": info["nnz"],
            "w_ij": info["w"],
            "max_abs_ij": info["max_abs"],
            "log_nnz_ij": float(np.log(info["nnz"] + 1)),
            "log_w_ij": float(np.log(info["w"] + 1e-12)),
            "r_nnz_ij": info["nnz"] / (nnz_ii + 1e-12),
            "r_w_ij": info["w"] / (sum_abs_if + 1e-12),
        })

    return node_feat, edge_feats


def main():
    if len(sys.argv) != 2:
        print("Uso: mpiexec -n N python3 extract_features_spe10.py <mat_id>")
        sys.exit(1)

    mat_id = sys.argv[1]  # ex: "10", "30", "105"
    matrix_file = f"matrices/mat_{mat_id}.petscbin"

    if not os.path.exists(matrix_file):
        print(f"Erro: {matrix_file} nao encontrado")
        sys.exit(1)

    comm = PETSc.COMM_WORLD
    mpi_comm = MPI.COMM_WORLD
    rank = comm.getRank()
    size = comm.getSize()

    # Carregar matriz binaria do PETSc (distribuida automaticamente)
    viewer = PETSc.Viewer().createBinary(matrix_file, mode='r', comm=comm)
    A = PETSc.Mat().load(viewer)
    viewer.destroy()
    n = A.getSize()[0]

    node_feat, edge_feats = compute_node_features(A, comm, n, rank, size)

    all_nodes = mpi_comm.gather(node_feat, root=0)
    all_edges = mpi_comm.gather(edge_feats, root=0)

    if rank == 0:
        edges_flat = []
        for sublist in all_edges:
            edges_flat.extend(sublist)

        df_nodes = pd.DataFrame(all_nodes)
        df_edges = pd.DataFrame(edges_flat)

        df_nodes.to_csv(f"features_nodes_mat_{mat_id}_p{size}.csv", index=False)
        df_edges.to_csv(f"features_edges_mat_{mat_id}_p{size}.csv", index=False)

        print(f"mat_{mat_id}: {len(df_nodes)} nos, {len(df_edges)} arestas")

    A.destroy()


if __name__ == "__main__":
    main()
