import sys
import time
import numpy as np
import torch
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

def load_rhs(rhs_path, n):
    data = np.fromfile(rhs_path, dtype=np.float64)
    if data.size == n:
        return data
    if data.size == n + 1:
        return data[1:]
    if data.size == n + 2:
        return data[2:]
    raise RuntimeError(f"RHS size mismatch: {data.size} doubles, expected {n}")

def run_gmres_one_varied(A, b, varied_rank, k_try, base_k=0, max_it=200, rtol=1e-6):
    """Executa GMRES com ASM onde um subdomínio usa k_try e os demais usam base_k.
       Retorna o número de iterações, o resíduo e o tempo de execução."""
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
        if i == varied_rank:
            subpc.setFactorLevels(k_try)
        else:
            subpc.setFactorLevels(base_k)
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
    if len(sys.argv) != 2:
        print("Uso: mpirun -np N python label_subdomains.py mat_XX.txt")
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
            print("Requer pelo menos 2 processos MPI.")
        sys.exit(1)

    rows, cols, vals, n = parse_dump(mat_file)
    A = build_distributed_A(PETSc.COMM_WORLD, rows, cols, vals, n)

    rhs_file = f"rhs_{mat_id}_1.bin"
    if not os.path.exists(rhs_file):
        if rank == 0:
            print(f"RHS {rhs_file} não encontrado.")
        sys.exit(1)
    b_full = load_rhs(rhs_file, n)

    b = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    b.setSizes([PETSc.DECIDE, n])
    b.setType(PETSc.Vec.Type.MPI)
    b.setUp()
    rstart, rend = b.getOwnershipRange()
    local_vals = b_full[rstart:rend]
    if len(local_vals) != (rend - rstart):
        raise RuntimeError(f"Rank {rank}: tamanho local {len(local_vals)} != esperado {rend - rstart}")
    b.setArray(local_vals)
    b.assemblyBegin()
    b.assemblyEnd()

    kmax = 3
    base_k = 0
    best_k_local = -1
    best_time_local = float('inf')

    for varied_rank in range(size):
        best_k = None
        best_time = float('inf')
        for k_try in range(kmax + 1):
            its, res, elapsed = run_gmres_one_varied(A, b, varied_rank, k_try, base_k, max_it=200, rtol=1e-6)
            if rank == 0:
                print(f"[Test] varied_rank={varied_rank}, k={k_try}, its={its}, time={elapsed:.6f} s")
            if elapsed < best_time:
                best_time = elapsed
                best_k = k_try
        if rank == varied_rank:
            best_k_local = best_k
            best_time_local = best_time
            print(f"[Rank {rank}] Melhor k = {best_k} com tempo {best_time:.6f} s")

    local_best = best_k_local if best_k_local != -1 else -1
    gathered = mpi_comm.gather(local_best, root=0)

    if rank == 0:
        labels = torch.tensor(gathered, dtype=torch.long)
        out_file = f"labels_{mat_id}_p{size}.pt"
        torch.save(labels, out_file)
        print(f"Labels saved: {out_file}")
        print(f"Labels: {labels.tolist()}")

    A.destroy()
    b.destroy()

if __name__ == "__main__":
    main()
