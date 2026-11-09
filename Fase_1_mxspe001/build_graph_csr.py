#!/usr/bin/env python3
# build_graph_csr.py
# Uso: mpirun -np N python build_graph_csr.py mat_XX.txt

import sys
import numpy as np
import torch
from torch_geometric.data import Data
from petsc4py import PETSc
from mpi4py import MPI
import re
import os

def parse_dump(txt_path):
    rows = []
    cols = []
    vals = []
    n = None
    re_dim = re.compile(r"#\s*Matrix\s+DIM\s*=\s*(\d+)\s*x\s*(\d+)")
    re_gi = re.compile(r"^\s*gi\s*=\s*(\d+)\s*$")
    re_gj = re.compile(r"^\s*gj\s*=\s*(\d+)\s*$")
    re_block = re.compile(r"^\s*block\s*=\s*\[\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\]\s*$")
    gi = None
    gj = None
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

def main():
    if len(sys.argv) != 2:
        print("Uso: mpirun -np N python build_graph_csr.py mat_XX.txt")
        sys.exit(1)
    mat_file = sys.argv[1]
    match = re.search(r"mat_(\d+)\.txt", mat_file)
    if not match:
        raise RuntimeError("Invalid matrix filename")
    mat_id = match.group(1)

    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()
    if size < 2:
        if rank == 0:
            print("Este script requer pelo menos 2 MPI processos.")
        sys.exit(1)

    rows, cols, vals, n = parse_dump(mat_file)
    A = build_distributed_A(PETSc.COMM_WORLD, rows, cols, vals, n)

    rstart, rend = A.getOwnershipRange()
    local_rows = rend - rstart
    degrees = []
    for row in range(rstart, rend):
        ret = A.getRow(row)
        if isinstance(ret, tuple):
            if len(ret) == 3:
                ncols = ret[0]
            elif len(ret) == 2:
                if isinstance(ret[0], (int, np.integer)):
                    ncols = ret[0]
                else:
                    ncols = len(ret[0])
            else:
                ncols = 0
        else:
            ncols = 0
        degrees.append(ncols)
    mean_degree = np.mean(degrees) if degrees else 0.0
    feat = np.array([local_rows, mean_degree, mean_degree / max(1, local_rows)], dtype=np.float32)

    all_feats = mpi_comm.gather(feat, root=0)

    neighbors = set()
    for row in range(rstart, rend):
        ret = A.getRow(row)
        if isinstance(ret, tuple):
            if len(ret) == 3:
                cols = ret[1]
            elif len(ret) == 2 and isinstance(ret[1], (list, tuple, np.ndarray)):
                cols = ret[1]
            else:
                continue
        else:
            continue
        for col in cols:
            owner = int(col * size // n)
            if owner != rank:
                neighbors.add(owner)
    all_neigh = mpi_comm.allgather(list(neighbors))

    if rank == 0:
        edges = []
        for i, neighs in enumerate(all_neigh):
            for j in neighs:
                if i < j:
                    edges.append((i, j))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        node_features = torch.tensor(np.stack(all_feats), dtype=torch.float)
        graph_data = Data(x=node_features, edge_index=edge_index)
        out_file = f"graph_{mat_id}_p{size}.pt"
        torch.save(graph_data, out_file)
        print(f"Graph saved: {out_file}")
        print(f"  Nodes: {node_features.shape[0]}, Features: {node_features.shape[1]}")
        if edge_index.numel() > 0:
            print(f"  Edges: {edge_index.shape[1]}")
        else:
            print("  Edges: 0")
        print(f"  Labels para info: {node_features[:,0].int().tolist()}")

    A.destroy()

if __name__ == "__main__":
    main()
