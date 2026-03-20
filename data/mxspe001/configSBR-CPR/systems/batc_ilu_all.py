import os
import re
import glob
import time
import numpy as np
from petsc4py import PETSc

def load_matrix_from_dump(dump_path):
    print(f"[LENDO MATRIZ] {os.path.basename(dump_path)}")

    n = None                # dimensão
    gi = None               # índice global da linha
    gj = None               # índice global da coluna
    nnz = 0                 # contador de elementos não-nulos
    A = None                # matriz PETSc

    with open(dump_path, "r") as f:
        for line in f:
            s = line.strip()

            # Linha com dimensão da matriz
            if s.startswith("#") and "Matrix DIM" in s:
                # Exemplo: "# Matrix DIM = 302x302"
                m = re.search(r"=\s*(\d+)x(\d+)", s)
                if m:
                    n = int(m.group(1))
                    # Cria matriz PETSc do tipo AIJ, sequencial
                    A = PETSc.Mat().create()
                    A.setSizes([n, n])
                    A.setType(PETSc.Mat.Type.AIJ)
                    A.setUp()

            # Índice global da linha
            elif s.startswith("gi ="):
                gi = int(s.split("=")[1])

            # Índice global da coluna (começo do bloco)
            elif s.startswith("gj ="):
                gj = int(s.split("=")[1])

            # Valores do bloco
            elif s.startswith("block ="):
                if A is None or n is None or gi is None or gj is None:
                    continue

                inside = s[s.find("[") + 1 : s.find("]")]
                if not inside.strip():
                    continue

                vals = [float(x) for x in inside.split()]

                for offset, val in enumerate(vals):
                    row = gi
                    col = gj + offset

                    # Segurança: checar limites
                    if not (0 <= row < n and 0 <= col < n):
                        raise RuntimeError(
                            f"Índice fora do intervalo ao montar a matriz: "
                            f"({row}, {col}) para n={n}"
                        )

                    A.setValue(row, col, val, addv=PETSc.InsertMode.ADD_VALUES)
                    nnz += 1

    if A is None or n is None:
        raise RuntimeError("Não foi possível criar a matriz a partir do dump (n ou A indefinidos).")

    A.assemble()

    print(f"  -> n = {n}, nnz (contados) = {nnz}")
    print("  -> Matriz PETSc montada.\n")

    return A, n, nnz

# Ler RHS binário rhs_XX_*.bin assumindo float64 com header
def load_rhs(rhs_path, n):
    print(f"[LENDO RHS] {os.path.basename(rhs_path)}")

    size_bytes = os.path.getsize(rhs_path)
    print(f"  -> tamanho do arquivo: {size_bytes} bytes")

    # Tentamos double (8 bytes)
    n_total_d = size_bytes // 8
    print(f"  -> número total de doubles possíveis: {n_total_d}")

    with open(rhs_path, "rb") as f:
        data = f.read()

    # Interpretar como float64
    arr = np.frombuffer(data, dtype=np.float64, count=n_total_d)

    # Casos típicos visto no sistema 10:
    # - n_total_d == n       → sem header
    # - n_total_d == n + 1   → 1 valor de header no início
    if n_total_d == n:
        vec_vals = arr
        print("  -> formato detectado: exatamente n valores (sem header).")
    elif n_total_d == n + 1:
        vec_vals = arr[1:]
        print("  -> formato detectado: n+1 valores (ignorando 1 header).")
    else:
        # fallback simples: pegar os últimos n valores
        if n_total_d > n:
            vec_vals = arr[-n:]
            print("  -> formato não padrão; usando últimos n valores.")
        else:
            raise RuntimeError(
                f"Arquivo {rhs_path} tem poucos valores ({n_total_d}) para n={n}."
            )

    if vec_vals.shape[0] != n:
        raise RuntimeError(
            f"Depois do tratamento, RHS ficou com {vec_vals.shape[0]} entradas (esperado {n})."
        )

    print(f"  -> RHS carregado com tamanho {n}.\n")

    # Cria Vec PETSc
    b = PETSc.Vec().createSeq(n)
    b.setArray(vec_vals.copy())  # copy para não usar o buffer direto
    return b


#Rodar GMRES + ILU(k) para uma matriz e um RHS
def run_ilu_sweep(A, b, ks=(0, 1, 2, 3)):
    results = []
    n = A.getSize()[0]

    print("=== VARREDURA DE ILU(k) ===")
    for k in ks:
        print(f"\n[ILU({k})] Rodando GMRES...")

        x = PETSc.Vec().createSeq(n)

        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        ksp.setType("gmres")

        pc = ksp.getPC()
        pc.setType("ilu")
        # nível de preenchimento k
        pc.setFactorLevels(k)

        ksp.setFromOptions()  # deixa usar -ksp_monitor etc se quiser

        t0 = time.perf_counter()
        ksp.solve(b, x)
        t1 = time.perf_counter()

        converged = ksp.is_converged
        reason = ksp.getConvergedReason()
        its = ksp.getIterationNumber()
        res = ksp.getResidualNorm()
        dt = t1 - t0

        print(f"  -> convergiu? {converged}")
        print(f"  -> reason   = {reason}")
        print(f"  -> iterações = {its}")
        print(f"  -> ||res||   = {res:.3e}")
        print(f"  -> tempo     = {dt:.4f} s")

        results.append({
            "k": k,
            "converged": bool(converged),
            "reason": int(reason),
            "its": int(its),
            "residual": float(res),
            "time": float(dt),
        })

        x.destroy()
        ksp.destroy()

    print("\n=== FIM VARREDURA ILU(k) ===\n")
    return results


# 4) Varre todos os sistemas da pasta
def main():
    # Vamos procurar todos os mat_XX.txt que existirem
    dump_files = sorted(glob.glob("mat_*.txt"))

    if not dump_files:
        print("Nenhum mat_XX.txt encontrado na pasta atual.")
        return

    # CSV de saída
    out_csv = "ilu_all_results.csv"
    print(f"Resultados serão salvos em: {out_csv}\n")

    with open(out_csv, "w") as f_out:
        f_out.write("matrix_id,rhs_file,n,nnz,k,converged,reason,its,residual,time\n")

        for dump_path in dump_files:
            base = os.path.basename(dump_path)
            # extrai o "XX" de mat_XX.txt
            m = re.match(r"mat_(\d+)\.txt", base)
            if not m:
                continue
            idx = m.group(1)

            # Carrega matriz
            A, n, nnz = load_matrix_from_dump(dump_path)

            # Procura RHS dessa matriz: rhs_XX_*.bin
            rhs_pattern = f"rhs_{idx}_*.bin"
            rhs_files = sorted(glob.glob(rhs_pattern))

            if not rhs_files:
                print(f"Nenhum RHS encontrado para matriz {base} (padrão {rhs_pattern}).\n")
                A.destroy()
                continue

            for rhs_path in rhs_files:
                # Carrega RHS
                b = load_rhs(rhs_path, n)

                # Varredura de ILU(k)
                results = run_ilu_sweep(A, b, ks=(0, 1, 2, 3))

                # Salva no CSV
                for r in results:
                    f_out.write(
                        f"{idx},{os.path.basename(rhs_path)},{n},{nnz},"
                        f"{r['k']},{r['converged']},{r['reason']},"
                        f"{r['its']},{r['residual']},{r['time']}\n"
                    )
                f_out.flush()

                b.destroy()

            A.destroy()

    print("\n Varredura de todas as matrizes finalizada!")
    print(f"Resultados consolidados em: {out_csv}")


if __name__ == "__main__":
    main()
