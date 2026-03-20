import os
import re
import csv
import tarfile
import time
import numpy as np

from petsc4py import PETSc
from scipy.sparse import coo_matrix
from scipy.io import mmwrite


def find_mat_ids():
    """
    Encontra IDs disponíveis a partir de mat_XX.tar.gz ou mat_XX.txt
    Retorna lista de inteiros [10, 30, ...]
    """
    ids = set()
    for fn in os.listdir("."):
        m = re.match(r"mat_(\d+)\.tar\.gz$", fn)
        if m:
            ids.add(int(m.group(1)))
        m = re.match(r"mat_(\d+)\.txt$", fn)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def extract_mat_txt(mat_id):
    """
    Extrai mat_{id}.tar.gz -> mat_{id}.txt (se ainda não existir)
    """
    tgz = f"mat_{mat_id}.tar.gz"
    txt = f"mat_{mat_id}.txt"

    if os.path.exists(txt):
        return txt

    if not os.path.exists(tgz):
        raise FileNotFoundError(f"Não achei {tgz} e {txt}.")

    with tarfile.open(tgz, "r:gz") as tar:
        members = tar.getmembers()
        target = None
        for mem in members:
            if mem.name.endswith(".txt"):
                target = mem
                break
        if target is None:
            raise RuntimeError(f"{tgz} não contém .txt (olhei membros do tar).")

        tar.extract(target, ".")
        extracted = target.name
        if extracted != txt:
            os.rename(extracted, txt)

    return txt


def parse_petsc_dump_to_coo(txt_path):
    """
    Parse do formato 'Matrix DUMP with global indexes'
    Exemplo do arquivo:
      gi = 0
        gj = 0
        block = [ 1.0 ]
        gj = 1
        block = [ -0.9 ]
      gi = 1
        gj = 0
        block = [ ... ]
    Retorna (rows, cols, vals, n)
    """
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
                    raise RuntimeError(f"Matriz não quadrada: {n1}x{n2}")
                n = n1
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
                v = float(m.group(1))
                rows.append(gi)
                cols.append(gj)
                vals.append(v)

    if n is None:
        raise RuntimeError(f"Não encontrei linha '# Matrix DIM = ...' em {txt_path}")

    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(vals, dtype=np.float64), n


def dump_to_mtx(txt_path, mtx_path):
    """
    Converte dump -> MatrixMarket .mtx via SciPy
    """
    rows, cols, vals, n = parse_petsc_dump_to_coo(txt_path)
    A = coo_matrix((vals, (rows, cols)), shape=(n, n))
    mmwrite(mtx_path, A)
    return n, A.nnz


def find_rhs_files(mat_id):
    """
    Lista rhs_{id}_*.bin existentes.
    """
    rhs = []
    pat = re.compile(rf"^rhs_{mat_id}_(\d+)\.bin$")
    for fn in os.listdir("."):
        m = pat.match(fn)
        if m:
            rhs.append((int(m.group(1)), fn))
    rhs.sort(key=lambda x: x[0])
    return [fn for _, fn in rhs]


def load_rhs_bin(rhs_path, n):
    """
    RHS binário: detecta float64 com header (n+1 doubles) ou sem header (n doubles)
    Retorna np.ndarray float64 tamanho n
    """
    size_bytes = os.path.getsize(rhs_path)

    # candidato float64
    if size_bytes % 8 != 0:
        raise RuntimeError(f"{rhs_path}: tamanho não é múltiplo de 8 bytes.")

    total_doubles = size_bytes // 8

    data = np.fromfile(rhs_path, dtype=np.float64)
    if data.size == n:
        return data
    if data.size == n + 1:
        # ignora header
        return data[1:]

    # fallback: às vezes pode ter 2 headers (raro) -> tenta n+2
    if data.size == n + 2:
        return data[2:]

    raise RuntimeError(
        f"{rhs_path}: não consegui ajustar RHS ao tamanho n={n}. "
        f"li {data.size} doubles."
    )


# PETSc: carregar MTX e resolver
from scipy.io import mmread
from scipy.sparse import coo_matrix

def load_petsc_mat_from_mtx(mtx_path):
    """
    Lê o .mtx com SciPy e monta uma Mat PETSc AIJ.
    (Evita MatLoad via Viewer ASCII, que não é suportado na sua build.)
    """
    M = mmread(mtx_path)

    # garantir COO
    if not isinstance(M, coo_matrix):
        M = M.tocoo()

    n, m = M.shape
    if n != m:
        raise RuntimeError(f"Matriz não quadrada no .mtx: {n}x{m}")

    A = PETSc.Mat().create()
    A.setSizes([n, n])
    A.setType(PETSc.Mat.Type.AIJ)
    A.setUp()

    # inserir valores
    for i, j, v in zip(M.row, M.col, M.data):
        A.setValue(int(i), int(j), float(v), addv=PETSc.InsertMode.ADD_VALUES)

    A.assemble()
    return A

def run_gmres_ilu(A, b_np, ilu_k, rtol=1e-8, max_it=500):
    """
    GMRES + PCILU(levels=ilu_k).
    Retorna dict com métricas.
    """
    b = PETSc.Vec().createWithArray(b_np)
    x = b.duplicate()
    x.set(0)

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=rtol, max_it=max_it)

    pc = ksp.getPC()
    pc.setType("ilu")
    pc.setFactorLevels(ilu_k)

    ksp.setFromOptions()

    t0 = time.perf_counter()
    ksp.solve(b, x)
    t1 = time.perf_counter()

    its = ksp.getIterationNumber()
    reason = int(ksp.getConvergedReason())
    converged = reason > 0

    # residual norm: PETSc pode fornecer via getResidualNorm()
    res = float(ksp.getResidualNorm())

    return {
        "converged": converged,
        "reason": reason,
        "its": its,
        "residual": res,
        "time_s": t1 - t0,
    }


def main():
    out_csv = "ilu_all_results.csv"
    print(f"Resultados serão salvos em: {out_csv}\n")

    mat_ids = find_mat_ids()
    if not mat_ids:
        raise RuntimeError("Não encontrei mat_*.tar.gz nem mat_*.txt na pasta.")

    # abre CSV (append se já existir)
    file_exists = os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "mat_id", "mat_txt", "mat_mtx",
                "rhs_file",
                "ilu_k", "converged", "reason", "its", "residual", "time_s"
            ],
        )
        if not file_exists:
            writer.writeheader()

        for mat_id in mat_ids:
            mat_txt = extract_mat_txt(mat_id)
            mat_mtx = f"mat_{mat_id}.mtx"

            print(f"[MATRIZ] mat_{mat_id}")
            if not os.path.exists(mat_mtx):
                n, nnz = dump_to_mtx(mat_txt, mat_mtx)
                print(f"  -> dump -> mtx OK: n={n}, nnz={nnz} ({mat_mtx})")
            else:
                # se já existe, só reporta
                print(f"  -> mtx já existe: {mat_mtx}")

            # carrega no PETSc
            A = load_petsc_mat_from_mtx(mat_mtx)
            n = A.getSize()[0]
            print(f"  -> PETSc load OK: n={n}, nnz~={A.getInfo()['nz_used']}")

            rhs_files = find_rhs_files(mat_id)
            if not rhs_files:
                print(f"  -> (sem RHS rhs_{mat_id}_*.bin) pulando.\n")
                continue

            for rhs in rhs_files:
                print(f"\n  [RHS] {rhs}")
                b_np = load_rhs_bin(rhs, n)
                print(f"    -> RHS OK: len={b_np.size}, dtype={b_np.dtype}")

                for k in [0, 1, 2, 3]:
                    print(f"    [ILU({k})] GMRES...")
                    r = run_gmres_ilu(A, b_np, ilu_k=k, rtol=1e-8, max_it=500)
                    print(
                        f"      conv={r['converged']} reason={r['reason']} "
                        f"its={r['its']} res={r['residual']:.3e} time={r['time_s']:.4f}s"
                    )

                    writer.writerow({
                        "mat_id": mat_id,
                        "mat_txt": mat_txt,
                        "mat_mtx": mat_mtx,
                        "rhs_file": rhs,
                        "ilu_k": k,
                        "converged": r["converged"],
                        "reason": r["reason"],
                        "its": r["its"],
                        "residual": r["residual"],
                        "time_s": r["time_s"],
                    })
                    fcsv.flush()

            print("\n")

    print("\nVarredura de todas as matrizes finalizada!")
    print(f"Resultados consolidados em: {out_csv}")


if __name__ == "__main__":
    main()
