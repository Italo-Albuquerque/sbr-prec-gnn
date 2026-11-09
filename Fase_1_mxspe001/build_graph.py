import sys
import numpy as np
import torch
from torch_geometric.data import Data
from petsc4py import PETSc
from mpi4py import MPI
import re
import os

#Funções de leitura do dump e construção da matriz PETSc
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

def get_subdomain_features(A, rstart, rend):
    """Retorna features: [tamanho local, grau médio, densidade média]"""
    local_n = rend - rstart
    total_degree = 0
    for row in range(rstart, rend):
        #Pegando o primeiro elemento que é o número de colunas
        row_info = A.getRow(row)
        if isinstance(row_info, tuple):
            # Tentando extrair o número de colunas
            if len(row_info) >= 1:
                # Se o primeiro elemento for um inteiro, é o ncols
                if isinstance(row_info[0], (int, np.integer)):
                    ncols = row_info[0]
                else:
                    # Se o primeiro elemento for uma lista, seu comprimento é ncols
                    ncols = len(row_info[0])
            else:
                ncols = 0
        else:
            ncols = 0
        total_degree += ncols
    mean_degree = total_degree / local_n if local_n > 0 else 0.0
    local_density = mean_degree / max(1, local_n)
    return np.array([local_n, mean_degree, local_density], dtype=np.float32)

def get_subdomain_edges(A, comm, size, n, rstart, rend):
    """Retorna lista de arestas (i,j) entre subdomínios baseado na partição contígua"""
    rank = comm.Get_rank()
    neighbors = set()
    for row in range(rstart, rend):
        row_info = A.getRow(row)
        # Extrai a lista de colunas
        if isinstance(row_info, tuple):
            if len(row_info) >= 2 and isinstance(row_info[1], (list, tuple, np.ndarray)):
                cols = row_info[1]
            elif len(row_info) >= 1 and isinstance(row_info[0], (list, tuple, np.ndarray)):
                cols = row_info[0]
            else:
                continue
        else:
            continue
        for col in cols:
            owner = int(col * size // n)
            if owner != rank:
                neighbors.add(owner)
    all_neigh = comm.allgather(list(neighbors))
    edges = []
    for i, neighs in enumerate(all_neigh):
        for j in neighs:
            if i < j:
                edges.append((i, j))
    return edges

def main():
    if len(sys.argv) != 2:
        print("Uso: mpirun -np N python build_graph.py mat_XX.txt")
        sys.exit(1)
    mat_file = sys.argv[1]
    match = re.search(r"mat_(\d+)\.txt", mat_file)
    if not match:
        raise RuntimeError("Invalid matrix filename")
    mat_id = match.group(1)

    # Uso mpi4py para comunicação
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    # Crio o comunicador PETSc a partir do mpi_comm
    petsc_comm = PETSc.Comm().tompi4py()  
    A = None
    try:
        # Carregar matriz 
        rows, cols, vals, n = parse_dump(mat_file)
        A = build_distributed_A(PETSc.COMM_WORLD, rows, cols, vals, n)

        rstart, rend = A.getOwnershipRange()

        # Features locais
        feat = get_subdomain_features(A, rstart, rend)

        # Reunir features no rank 0 via mpi4py
        all_feats = mpi_comm.gather(feat, root=0)

        if rank == 0:
            # Construir arestas (precisa do A, que está distribuído)
            edges = get_subdomain_edges(A, mpi_comm, size, n, rstart, rend)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            node_features = torch.tensor(np.stack(all_feats), dtype=torch.float)
            graph_data = Data(x=node_features, edge_index=edge_index)
            out_file = f"graph_{mat_id}_p{size}.pt"
            torch.save(graph_data, out_file)
            print(f"Graph saved: {out_file}")
            print(f"  Nodes: {node_features.shape[0]}, Features: {node_features.shape[1]}")
            print(f"  Edges: {edge_index.shape[1]}")
    finally:
        if A is not None:
            A.destroy()

if __name__ == "__main__":
    main()
