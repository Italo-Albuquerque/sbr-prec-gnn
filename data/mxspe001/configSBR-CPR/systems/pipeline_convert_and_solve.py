import os
import glob
import re
import csv
import numpy as np
from scipy.io import mmwrite, mmread
from scipy.sparse import coo_matrix
from petsc4py import PETSc

# Diretório atual: systems
BASE_DIR = os.getcwd()

RESULTS_CSV = os.path.join(BASE_DIR, "results_gmres.csv")

def csv_to_mtx(csv_path, mtx_path):
    print(f"\n[CSV→MTX] Lendo {csv_path}")
    data = np.loadtxt(csv_path)

    rows = data[:, 0].astype(int)
    cols = data[:, 1].astype(int)
    vals = data[:, 2]

    # se índices começam em 1, converte para 0-based
    if rows.min() == 1 or cols.min() == 1:
        rows -= 1
        cols -= 1

    n = max(rows.max(), cols.max()) + 1
    A = coo_matrix((vals, (rows, cols)), shape=(n, n))

    print(f"    matriz {n}x{n}, nnz = {A.nnz}")
    mmwrite(mtx_path, A)
    print(f"    salvo em {mtx_path}")
    return n

def mtx_to_petsc_bin(mtx_path, bin_path):
    print(f"[MTX→PETSc] Lendo {mtx_path}")
    A_scipy = mmread(mtx_path).tocsr()
    n, m = A_scipy.shape
    print(f"    matriz {n}x{m}, nnz = {A_scipy.nnz}")

    ia = A_scipy.indptr.astype(PETSc.IntType)
    ja = A_scipy.indices.astype(PETSc.IntType)
    a  = A_scipy.data

    A = PETSc.Mat().createAIJ(size=A_scipy.shape, csr=(ia, ja, a))
    A.assemble()

    viewer = PETSc.Viewer().createBinary(bin_path, 'w')
    A.view(viewer)
    viewer.destroy()
    print(f"    salvo em {bin_path}")

def run_gmres(mat_bin, rhs_bin):
    print(f"[GMRES] A={os.path.basename(mat_bin)}, b={os.path.basename(rhs_bin)}")

    A = PETSc.Mat().load(PETSc.Viewer().createBinary(mat_bin, 'r'))
    b = PETSc.Vec().load(PETSc.Viewer().createBinary(rhs_bin, 'r'))
    x = b.duplicate()

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType("gmres")

    pc = ksp.getPC()
    pc.setType("ilu")
    pc.setFactorLevels(0)

    ksp.setTolerances(rtol=1e-8)
    ksp.setFromOptions()

    ksp.solve(b, x)

    its = ksp.getIterationNumber()
    res = ksp.getResidualNorm()
    reason = ksp.getConvergedReason()

    print(f"    its = {its}, res = {res:.3e}, reason = {reason}")
    return its, res, reason

def find_rhs_for_mat(mat_csv_name):
    # extrai o número XX do "mat_XX.csv"
    m = re.match(r"mat_(\d+)\.csv", mat_csv_name)
    if not m:
        return None
    idx = m.group(1)
    pattern = os.path.join(BASE_DIR, f"rhs_{idx}_*.bin")
    rhs_candidates = sorted(glob.glob(pattern))
    return rhs_candidates[0] if rhs_candidates else None

def main():
    mat_csv_files = sorted(glob.glob(os.path.join(BASE_DIR, "mat_*.csv")))
    if not mat_csv_files:
        print("Nenhum mat_*.csv encontrado.")
        return

    # prepara arquivo de resultados
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["matrix", "rhs", "n", "its", "residual", "reason"])

    for csv_path in mat_csv_files:
        csv_name = os.path.basename(csv_path)
        rhs_path = find_rhs_for_mat(csv_name)

        if rhs_path is None:
            print(f"\n[SKIP] {csv_name} → nenhum rhs_XX_*.bin correspondente encontrado.")
            continue

        idx = re.match(r"mat_(\d+)\.csv", csv_name).group(1)

        mtx_path = os.path.join(BASE_DIR, f"mat_{idx}.mtx")
        bin_path = os.path.join(BASE_DIR, f"mat_{idx}.bin")

        # 1) CSV → MTX (se ainda não existir)
        if not os.path.exists(mtx_path):
            n = csv_to_mtx(csv_path, mtx_path)
        else:
            print(f"\n[INFO] {mtx_path} já existe. Pulando conversão CSV→MTX.")
            # n pode ser lido depois, mas não é crítico aqui
            n = None

        # 2) MTX → PETSc bin (se ainda não existir)
        if not os.path.exists(bin_path):
            mtx_to_petsc_bin(mtx_path, bin_path)

        # 3) Rodar GMRES
        its, res, reason = run_gmres(bin_path, rhs_path)

        # salva resultado
        with open(RESULTS_CSV, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([os.path.basename(bin_path),
                        os.path.basename(rhs_path),
                        "" if n is None else n,
                        its, res, reason])

if __name__ == "__main__":
    main()
