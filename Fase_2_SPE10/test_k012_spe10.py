import sys
import os
import time
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI


def run_one_k(A, b, k, max_it=1000, rtol=1e-3, atol=1e-50):
    """
    Roda GMRES + ASM (Block Jacobi) + ILU(k) em todos os subdominios.
    Retorna (its, res, reason, elapsed_max) onde elapsed_max e o tempo
    maximo entre todos os ranks MPI (gargalo real do solve).
    """
    ksp = PETSc.KSP().create(comm=A.getComm())
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)

    pc = ksp.getPC()
    pc.setType("asm")           # Additive Schwarz = Block Jacobi
    pc.setASMOverlap(0)
    pc.setUp()

    subksps = pc.getASMSubKSP()
    if isinstance(subksps, tuple):
        n_local, subksps = subksps
    for subksp in subksps:
        subpc = subksp.getPC()
        subpc.setType("ilu")
        subpc.setFactorLevels(k)

    x = b.duplicate()
    x.set(0.0)

    # Timer apenas ao redor do solve (solve_seconds)
    t0 = time.perf_counter()
    ksp.solve(b, x)
    t1 = time.perf_counter()
    elapsed = t1 - t0

    # Tempo maximo entre todos os ranks
    mpi_comm = MPI.COMM_WORLD
    elapsed_max = mpi_comm.allreduce(elapsed, op=MPI.MAX)

    its = ksp.getIterationNumber()
    res = ksp.getResidualNorm()
    reason = ksp.getConvergedReason()
    ksp.destroy()

    return its, res, reason, elapsed_max


def main():
    if len(sys.argv) != 3:
        print("Uso: mpiexec -n N python3 test_k012_spe10.py <mat_id> <rhs_filename>")
        print("Exemplo: mpiexec -n 4 python3 test_k012_spe10.py 10 rhs_10_1.petscbin")
        sys.exit(1)

    mat_id = sys.argv[1]
    rhs_name = sys.argv[2]

    matrix_file = f"matrices/mat_{mat_id}.petscbin"
    rhs_file = f"rhs/{rhs_name}"

    if not os.path.exists(matrix_file):
        print(f"Erro: {matrix_file} nao encontrado")
        sys.exit(1)
    if not os.path.exists(rhs_file):
        print(f"Erro: {rhs_file} nao encontrado")
        sys.exit(1)

    comm = PETSc.COMM_WORLD
    rank = comm.getRank()

    # Carregar matriz binaria via Viewer
    viewer_A = PETSc.Viewer().createBinary(matrix_file, mode='r', comm=comm)
    A = PETSc.Mat().load(viewer_A)
    viewer_A.destroy()
    n = A.getSize()[0]

    # Carregar RHS binario via Viewer
    viewer_b = PETSc.Viewer().createBinary(rhs_file, mode='r', comm=comm)
    b = PETSc.Vec().load(viewer_b)
    viewer_b.destroy()

    for k in [0, 1, 2]:
        its, res, reason, elapsed = run_one_k(A, b, k)
        converged = 1 if reason > 0 else 0
        status = "OK" if converged else "FALHOU"
        if rank == 0:
            print(f"{mat_id},{rhs_name},{k},{n},{its},{res:.6e},{reason},{elapsed:.3f},{status}")

    A.destroy()
    b.destroy()


if __name__ == "__main__":
    main()
