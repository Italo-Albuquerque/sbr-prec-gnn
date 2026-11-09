import os
import sys
import time
import itertools
import numpy as np
import pandas as pd
from petsc4py import PETSc
from mpi4py import MPI


def run_solver(A, b, k_per_rank, max_it=1000, rtol=1e-3, atol=1e-50):
    ksp = PETSc.KSP().create(comm=A.getComm())
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    pc = ksp.getPC()
    pc.setType("asm")
    pc.setASMOverlap(0)
    pc.setUp()
    subksps = pc.getASMSubKSP()
    if isinstance(subksps, tuple):
        n_local, subksps = subksps
    else:
        n_local = len(subksps)
    for i, subksp in enumerate(subksps):
        subpc = subksp.getPC()
        subpc.setType("ilu")
        subpc.setFactorLevels(int(k_per_rank[i]))
    x = b.duplicate()
    x.set(0.0)

    t0 = time.perf_counter()
    ksp.solve(b, x)
    t1 = time.perf_counter()

    its = ksp.getIterationNumber()
    res = ksp.getResidualNorm()
    reason = ksp.getConvergedReason()
    ksp.destroy()
    return its, res, reason, t1 - t0


def save_incremental(all_elite_rows, all_soft_labels):
    pd.DataFrame(all_soft_labels).to_csv("teacher_soft_labels.csv", index=False)
    pd.DataFrame(all_elite_rows).to_csv("teacher_all_configs.csv", index=False)


def main():
    comm = PETSc.COMM_WORLD
    mpi_comm = MPI.COMM_WORLD
    rank = comm.getRank()
    size = comm.getSize()

    if not os.path.exists("cluster_assignments.csv"):
        if rank == 0:
            print("cluster_assignments.csv nao encontrado.")
        sys.exit(1)

    df_clusters = pd.read_csv("cluster_assignments.csv")

    if len(sys.argv) > 1:
        mat_ids = [int(sys.argv[1])]
        if rank == 0:
            print(f"Processando apenas mat_{mat_ids[0]} com {size} processos")
    else:
        mat_ids = sorted(df_clusters["matrix_id"].unique())
        if rank == 0:
            print(f"Processando todas as {len(mat_ids)} matrizes com {size} processos")

    all_elite_rows = []
    all_soft_labels = []
    if os.path.exists("teacher_all_configs.csv"):
        all_elite_rows = pd.read_csv("teacher_all_configs.csv").to_dict("records")
    if os.path.exists("teacher_soft_labels.csv"):
        all_soft_labels = pd.read_csv("teacher_soft_labels.csv").to_dict("records")

    for mat_id in mat_ids:
        matrix_file = f"matrices/mat_{mat_id}.petscbin"

        # Pega os 3 primeiros RHS disponiveis para essa matriz
        rhs_candidates = sorted([f for f in os.listdir("rhs") if f.startswith(f"rhs_{mat_id}_")])
        if not rhs_candidates:
            if rank == 0:
                print(f"  Pulando mat_{mat_id}: nenhum RHS encontrado")
            continue
        rhs_to_use = rhs_candidates[:3]

        if not os.path.exists(matrix_file):
            if rank == 0:
                print(f"  Pulando mat_{mat_id}: {matrix_file} nao encontrado")
            continue

        df_mat = df_clusters[df_clusters["matrix_id"] == mat_id].sort_values("rank")
        if len(df_mat) != size:
            if rank == 0:
                print(f"  Pulando mat_{mat_id}: {len(df_mat)} nos mas {size} processos")
            continue

        n_clusters = int(df_mat["cluster"].max()) + 1

        if rank == 0:
            print(f"\n=== mat_{mat_id} ({n_clusters} clusters) ===")
            print(f"  RHSs: {rhs_to_use}")

        # Carregar matriz binaria
        viewer_A = PETSc.Viewer().createBinary(matrix_file, mode='r', comm=comm)
        A = PETSc.Mat().load(viewer_A)
        viewer_A.destroy()
        n = A.getSize()[0]

        # Carregar os 3 RHS na memoria de uma vez
        b_vecs = []
        for rhs_cand in rhs_to_use:
            rhs_path = os.path.join("rhs", rhs_cand)
            viewer_b = PETSc.Viewer().createBinary(rhs_path, mode='r', comm=comm)
            b_vecs.append(PETSc.Vec().load(viewer_b))
            viewer_b.destroy()

        all_configs = list(itertools.product([0, 1, 2], repeat=n_clusters))

        if rank == 0:
            print(f"  Testando {len(all_configs)} configuracoes (media de 3 RHS)")

        config_results = []
        for config in all_configs:
            k_per_rank = np.zeros(size, dtype=int)
            for _, row in df_mat.iterrows():
                k_per_rank[int(row["rank"])] = config[int(row["cluster"])]

            tempos = []
            its_sum = 0
            res_sum = 0
            all_converged = True

            # Roda o solver para os 3 RHS
            for b_temp in b_vecs:
                its, res, reason, elapsed = run_solver(A, b_temp, k_per_rank)
                elapsed_max = mpi_comm.allreduce(elapsed, op=MPI.MAX)

                tempos.append(elapsed_max)
                its_sum += its
                res_sum += res
                if reason <= 0:
                    all_converged = False

            # Calcula a media dos resultados
            elapsed = sum(tempos) / len(tempos)
            its = int(its_sum / len(b_vecs))
            res = res_sum / len(b_vecs)
            reason = 2 if all_converged else -3
            converged = all_converged

            config_results.append({
                "config": config,
                "its": its,
                "residual": res,
                "reason": reason,
                "converged": converged,
                "time": elapsed,
            })

            if rank == 0:
                status = "OK" if converged else "FALHOU"
                print(f"  config={config} its={its} res={res:.2e} time={elapsed:.2f}s {status}")

        if rank == 0:
            converged_configs = [c for c in config_results if c["converged"]]
            if not converged_configs:
                print(f"  Nenhuma configuracao convergiu")
                A.destroy()
                for b_temp in b_vecs:
                    b_temp.destroy()
                continue

            best = min(converged_configs, key=lambda x: x["time"])
            t_min = best["time"]
            threshold = 1.05 * t_min
            elite = [c for c in converged_configs if c["time"] <= threshold]

            print(f"  Melhor tempo: {t_min:.2f}s (config={best['config']})")
            print(f"  Elite: {len(elite)} configuracoes")

            elite_configs = [c["config"] for c in elite]
            for g in range(n_clusters):
                pi_g = np.zeros(3)
                for config in elite_configs:
                    pi_g[config[g]] += 1
                pi_g /= len(elite_configs)

                q_g1 = pi_g[1] + pi_g[2]
                q_g2 = pi_g[2]

                all_soft_labels.append({
                    "matrix_id": int(mat_id),
                    "cluster": g,
                    "pi_k0": pi_g[0],
                    "pi_k1": pi_g[1],
                    "pi_k2": pi_g[2],
                    "q_g1": q_g1,
                    "q_g2": q_g2,
                })

            for c in config_results:
                all_elite_rows.append({
                    "matrix_id": int(mat_id),
                    "config": str(c["config"]),
                    "its": c["its"],
                    "residual": c["residual"],
                    "converged": c["converged"],
                    "time": c["time"],
                    "in_elite": c in elite,
                })

            save_incremental(all_elite_rows, all_soft_labels)
            print(f"  Salvamento intermediario apos mat_{mat_id}")

        A.destroy()
        for b_temp in b_vecs:
            b_temp.destroy()

    if rank == 0:
        print(f"\nConcluido. Resultados em teacher_soft_labels.csv")


if __name__ == "__main__":
    main()
