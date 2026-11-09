#!/usr/bin/env python3
# collect_gnn_data.py
# Uso: mpirun -np N python collect_gnn_data.py mat_XX.txt

import sys
import numpy as np
import torch
from torch_geometric.data import Data
from petsc4py import PETSc
import re
import os

# ------------------------------------------------------------
# Funções auxiliares (parse dump, load rhs, build matrix)
# ------------------------------------------------------------
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

def load_rhs(rhs_path, n):
    data = np.fromfile(rhs_path, dtype=np.float64)
    if data.size == n:
        return data
    if data.size == n+1:
        return data[1:]
    if data.size == n+2:
        return data[2:]
    raise RuntimeError(f"RHS size mismatch: {data.size} doubles (n={n})")

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

def get_subdomain_features(A, comm):
    """Extrai features por processo (subdomínio)"""
    rank = comm.getRank()
    size = comm.getSize()
    rstart, rend = A.getOwnershipRange()
    local_n = rend - rstart
    row_degrees = []
    diag_dominance = []
    for row in range(rstart, rend):
        ncols, cols, vals = A.getRow(row)
        row_degrees.append(ncols)
        diag = None
        for j, v in zip(cols, vals):
            if j == row:
                diag = abs(v)
                break
        if diag is not None and diag > 0:
            off_diag_sum = sum(abs(v) for j, v in zip(cols, vals) if j != row)
            diag_dominance.append(diag / (off_diag_sum + 1e-12))
        else:
            diag_dominance.append(0.0)
    mean_degree = np.mean(row_degrees) if row_degrees else 0.0
    mean_diag_dom = np.mean(diag_dominance) if diag_dominance else 0.0
    features = torch.tensor([local_n, mean_degree, mean_diag_dom], dtype=torch.float)
    return features

def get_subdomain_graph(A, comm):
    """Retorna arestas entre subdomínios baseado na partição contígua"""
    rank = comm.getRank()
    size = comm.getSize()
    n = A.getSize()[0]
    rstart, rend = A.getOwnershipRange()
    neighbors = set()
    for row in range(rstart, rend):
        ncols, cols, vals = A.getRow(row)
        for col in cols:
            owner = int(col * size // n)
            if owner != rank:
                neighbors.add(owner)
    all_neighbors = comm.allgather(list(neighbors))
    edges = []
    for i, neighs in enumerate(all_neighbors):
        for j in neighs:
            if i < j:
                edges.append((i, j))
    return edges

def run_gmres_for_subdomain_k(A, b, subdomain_rank, k_try, base_k=0, max_it=200, rtol=1e-6):
    """Resolve com GMRES + ASM (overlap=0) onde um subdomínio usa ILU(k_try) e os demais usam base_k"""
    ksp = PETSc.KSP().create(comm=A.getComm())
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=rtol, max_it=max_it)
    pc = ksp.getPC()
    pc.setType("asm")               # Additive Schwarz, overlap 0 (equivale a BJACOBI)
    pc.setASMOverlap(0)             # sem sobreposição
    pc.setUp()                      # necessário para criar sub-KSPs
    # Obtém os sub-KSPs
    n_local, subksps = pc.getASMSubKSP()
    for i, subksp in enumerate(subksps):
        subpc = subksp.getPC()
        subpc.setType("ilu")
        if i == subdomain_rank:
            subpc.setFactorLevels(k_try)
        else:
            subpc.setFactorLevels(base_k)
    ksp.setFromOptions()
    x = b.duplicate()
    x.set(0.0)
    ksp.solve(b, x)
    its = ksp.getIterationNumber()
    res = ksp.getResidualNorm()
    ksp.destroy()
    return its, res

def main():
    if len(sys.argv) != 2:
        print("Uso: mpirun -np N python collect_gnn_data.py mat_XX.txt")
        sys.exit(1)
    mat_file = sys.argv[1]
    match = re.search(r"mat_(\d+)\.txt", mat_file)
    if not match:
        raise RuntimeError("Invalid matrix filename")
    mat_id = match.group(1)

    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    size = comm.getSize()
    if size < 2:
        if rank == 0:
            print("Este script requer pelo menos 2 processos MPI.")
        sys.exit(1)

    # 1. Carregar matriz
    rows, cols, vals, n = parse_dump(mat_file)
    A = build_distributed_A(comm, rows, cols, vals, n)

    # 2. Carregar RHS (primeiro RHS) e criar vetor distribuído
    rhs_file = f"rhs_{mat_id}_1.bin"
    if not os.path.exists(rhs_file):
        if rank == 0:
            print(f"Arquivo RHS {rhs_file} não encontrado. Pulando.")
        sys.exit(1)
    b_full = load_rhs(rhs_file, n)

    b = PETSc.Vec().create(comm=comm)
    b.setSizes([n, PETSc.DECIDE])
    b.setUp()
    rstart, rend = b.getOwnershipRange()
    local_vals = b_full[rstart:rend]
    b.setArray(local_vals)   # mais simples que setValues
    b.assemblyBegin()
    b.assemblyEnd()

    # 3. Para cada processo (subdomínio), testar k de 0 a kmax
    kmax = 3
    base_k = 0
    best_k_local = -1
    best_its = float('inf')
    for k_try in range(kmax+1):
        its, res = run_gmres_for_subdomain_k(A, b, rank, k_try, base_k, max_it=200, rtol=1e-6)
        print(f"[Rank {rank}] Test k={k_try} -> its={its}, res={res:.3e}")
        if its < best_its:
            best_its = its
            best_k_local = k_try

    # 4. Reunir dados no rank 0
    all_best_k = comm.gather(best_k_local, root=0)
    all_features = comm.gather(get_subdomain_features(A, comm), root=0)

    if rank == 0:
        edges = get_subdomain_graph(A, comm)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        node_features = torch.stack(all_features)
        y = torch.tensor(all_best_k, dtype=torch.long)
        graph_data = Data(x=node_features, edge_index=edge_index, y=y)
        out_file = f"gnn_data_{mat_id}.pt"
        torch.save(graph_data, out_file)
        print(f"Dados salvos em {out_file}")
        print(f"  Nós: {node_features.shape[0]}, Features: {node_features.shape[1]}")
        print(f"  Arestas: {edge_index.shape[1]}")
        print(f"  Rótulos: {y.tolist()}")

    A.destroy()
    b.destroy()

if __name__ == "__main__":
    main()
