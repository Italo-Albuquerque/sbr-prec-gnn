import os
import re
import time
import numpy as np
import torch
from petsc4py import PETSc
from mpi4py import MPI
import glob

def parse_dump(txt_path):
    rows, cols, vals = [], [], []
    n = None
    re_dim = re.compile(r"#\s*Matrix\s+DIM\s*=\s*(\d+)\s*x\s*(\d+)")
    re_gi = re.compile(r"^\s*gi\s*=\s*(\d+)\s*$")
    re_gj = re.compile(r"^\s*gj\s*=\s*(\d+)\s*$")
    re_block = re.compile(r"^\s*block\s*=\s*\[\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\]\s*$")
    gi = gj = None
    with open(txt_path, 'r') as f:
        for line in f:
            m = re_dim.search(line)
            if m:
                n = int(m.group(1))
                continue
            m = re_gi.match(line)
            if m:
                gi = int(m.group(1))
                continue
            m = re_gj.match(line)
            if m:
                gj = int(m.group(1))
                continue
            m = re_block.match(line)
            if m and gi is not None and gj is not None:
                rows.append(gi)
                cols.append(gj)
                vals.append(float(m.group(1)))
    if n is None:
        raise RuntimeError("DIM not found")
    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(vals, dtype=np.float64), n

def build_distributed_A(comm, rows, cols, vals, n):
    A = PETSc.Mat().create(comm=comm)
    A.setSizes([n, n])
    A.setType(PETSc.Mat.Type.AIJ)
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    mask = (rows >= rstart) & (rows < rend)
    for i, j, v in zip(rows[mask], cols[mask], vals[mask]):
        A.setValue(i, j, v, addv=PETSc.InsertMode.ADD_VALUES)
    A.assemblyBegin()
    A.assemblyEnd()
    return A

def load_rhs(rhs_path, n):
    data = np.fromfile(rhs_path, dtype=np.float64)
    if data.size == n:
        return data
    if data.size == n + 1:
        return data[1:]
    if data.size == n + 2:
        return data[2:]
    raise RuntimeError(f"RHS size mismatch: {data.size} doubles, expected {n}")

def run_gmres_with_local_k(A, b, k_local, max_it=200, rtol=1e-6):
    """Executa GMRES com ASM usando os níveis de k fornecidos por subdomínio."""
    ksp = PETSc.KSP().create(comm=A.getComm())
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=rtol, max_it=max_it)
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
        if i < len(k_local):
            subpc.setFactorLevels(k_local[i])
        else:
            subpc.setFactorLevels(0)  # fallback
    ksp.setFromOptions()
    x = b.duplicate()
    x.set(0.0)
    t0 = time.perf_counter()
    ksp.solve(b, x)
    t1 = time.perf_counter()
    its = ksp.getIterationNumber()
    res = ksp.getResidualNorm()
    ksp.destroy()
    return its, res, t1 - t0

def main():
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    size = comm.getSize()
    if size != 2:
        if rank == 0:
            print("Este script deve ser executado com exatamente 2 processos MPI.")
        sys.exit(1)

    # Encontra todas as matrizes
    mat_files = sorted(glob.glob("mat_*.txt"))
    if not mat_files:
        if rank == 0:
            print("Nenhum arquivo mat_*.txt encontrado.")
        sys.exit(1)

    results = []
    for mat_file in mat_files:
        match = re.search(r"mat_(\d+)\.txt", mat_file)
        if not match:
            continue
        mat_id = match.group(1)

        # Carrega os rótulos locais (arquivo .pt)
        label_file = f"labels_{mat_id}_p{size}.pt"
        if not os.path.exists(label_file):
            if rank == 0:
                print(f"Arquivo {label_file} não encontrado para {mat_file}. Pulando.")
            continue

        # Carrega os rótulos: lista de k por subdomínio (tamanho = size)
        labels = torch.load(label_file, weights_only=False).numpy()
        if len(labels) != size:
            if rank == 0:
                print(f"Tamanho dos rótulos ({len(labels)}) não coincide com número de processos ({size}). Pulando.")
            continue

        # Carrega a matriz
        rows, cols, vals, n = parse_dump(mat_file)
        A = build_distributed_A(PETSc.COMM_WORLD, rows, cols, vals, n)

        # Carrega o primeiro RHS
        rhs_file = f"rhs_{mat_id}_1.bin"
        if not os.path.exists(rhs_file):
            if rank == 0:
                print(f"RHS {rhs_file} não encontrado para {mat_file}. Pulando.")
            A.destroy()
            continue
        b_full = load_rhs(rhs_file, n)
        b = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        b.setSizes([PETSc.DECIDE, n])
        b.setType(PETSc.Vec.Type.MPI)
        b.setUp()
        rstart, rend = b.getOwnershipRange()
        b.setArray(b_full[rstart:rend])
        b.assemblyBegin()
        b.assemblyEnd()

        # Executa com os k locais
        its, res, t = run_gmres_with_local_k(A, b, labels, max_it=200, rtol=1e-6)

        if rank == 0:
            print(f"Matriz {mat_id}: k_locais={labels.tolist()}, its={its}, res={res:.3e}, time={t:.6f} s")
            results.append({
                "matrix_id": int(mat_id),
                "k0": int(labels[0]) if size > 0 else -1,
                "k1": int(labels[1]) if size > 1 else -1,
                "its": its,
                "residual": res,
                "time": t
            })

        A.destroy()
        b.destroy()

    if rank == 0:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv("local_times.csv", index=False)
        print("\nResultados salvos em local_times.csv")
        print(df)

if __name__ == "__main__":
    main()
