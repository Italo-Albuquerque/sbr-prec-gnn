from petsc4py import PETSc
import numpy as np
import re
import os

MAT_FILE = "mat_10.txt"
RHS_FILE = "rhs_10_1.bin"

def parse_dump(txt_path):
    n = None
    rows, cols, vals = [], [], []
    re_dim = re.compile(r"#\s*Matrix\s+DIM\s*=\s*(\d+)\s*x\s*(\d+)")
    re_gi = re.compile(r"^\s*gi\s*=\s*(\d+)\s*$")
    re_gj = re.compile(r"^\s*gj\s*=\s*(\d+)\s*$")
    re_block = re.compile(r"^\s*block\s*=\s*\[\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\]\s*$")

    gi = None
    gj = None
    with open(txt_path, "r") as f:
        for line in f:
            m = re_dim.search(line)
            if m:
                n1, n2 = int(m.group(1)), int(m.group(2))
                if n1 != n2:
                    raise RuntimeError(f"Não quadrada: {n1}x{n2}")
                n = n1
                continue
            m = re_gi.match(line); 
            if m: gi = int(m.group(1)); continue
            m = re_gj.match(line); 
            if m: gj = int(m.group(1)); continue
            m = re_block.match(line)
            if m and gi is not None and gj is not None:
                rows.append(gi)
                cols.append(gj)
                vals.append(float(m.group(1)))
    if n is None:
        raise RuntimeError("Não achei '# Matrix DIM = ...'")
    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(vals, dtype=np.float64), n

def load_rhs(rhs_path, n):
    data = np.fromfile(rhs_path, dtype=np.float64)
    if data.size == n:
        return data
    if data.size == n + 1:
        return data[1:]  # header
    if data.size == n + 2:
        return data[2:]
    raise RuntimeError(f"RHS {rhs_path}: li {data.size} doubles, esperado n ou n+1 (n={n}).")

def build_distributed_A(comm, rows, cols, vals, n):
    # Matriz distribuída por linhas
    A = PETSc.Mat().create(comm=comm)
    A.setType(PETSc.Mat.Type.AIJ)
    A.setSizes(([PETSc.DECIDE, n], [PETSc.DECIDE, n]))
    A.setUp()

    rstart, rend = A.getOwnershipRange()

    # Cada rank insere só as entradas cujas linhas pertencem ao seu range
    mask = (rows >= rstart) & (rows < rend)
    for i, j, v in zip(rows[mask], cols[mask], vals[mask]):
        A.setValue(int(i), int(j), float(v), addv=PETSc.InsertMode.ADD_VALUES)

    A.assemblyBegin()
    A.assemblyEnd()
    return A

def build_distributed_b(comm, b_full):
    n = b_full.size
    b = PETSc.Vec().create(comm=comm)
    b.setSizes((PETSc.DECIDE, n))
    b.setUp()

    rstart, rend = b.getOwnershipRange()
    b.setValues(range(rstart, rend), b_full[rstart:rend])
    b.assemblyBegin()
    b.assemblyEnd()
    return b

def main():
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()

    # Todo mundo lê o dump (simples e robusto); cada rank insere só suas linhas.
    rows, cols, vals, n = parse_dump(MAT_FILE)

    # RHS: todo mundo lê e depois distribui (barato para n pequeno; depois otimizamos)
    b_full = load_rhs(RHS_FILE, n)

    A = build_distributed_A(comm, rows, cols, vals, n)
    b = build_distributed_b(comm, b_full)
    x = b.duplicate()
    x.set(0)

    # KSP paralelo
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(A)
    ksp.setType("gmres")

    # Precondicionador paralelo: Block Jacobi, com ILU local em cada bloco
    pc = ksp.getPC()
    pc.setType("bjacobi")

    # IMPORTANTE: subsolvers configurados via options; usamos setFromOptions
    ksp.setFromOptions()

    ksp.solve(b, x)

    reason = ksp.getConvergedReason()
    its = ksp.getIterationNumber()
    res = ksp.getResidualNorm()

    if rank == 0:
        print("\n=== RESULTADO MPI ===")
        print("convergiu?", reason > 0)
        print("reason   =", int(reason))
        print("its      =", int(its))
        print("residual =", float(res))
        print("=====================\n")

if __name__ == "__main__":
    main()
