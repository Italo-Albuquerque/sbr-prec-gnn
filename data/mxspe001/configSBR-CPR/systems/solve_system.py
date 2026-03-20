from petsc4py import PETSc
import numpy as np
import os

# ---------------------------------------------------------
# 1. Carregar matriz A a partir do dump (mat_10.txt)
# ---------------------------------------------------------

dump_file = "mat_10.txt"
print(f"Lendo matriz do arquivo dump: {dump_file}")

rows = []
cols = []
vals = []

with open(dump_file, "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("gi ="):
            gi = int(line.split("=")[1].strip())
        elif line.startswith("gj ="):
            gj = int(line.split("=")[1].strip())
        elif line.startswith("block ="):
            block = float(line.split("[")[1].split("]")[0])
            rows.append(gi)
            cols.append(gj)
            vals.append(block)

n = max(max(rows), max(cols)) + 1
nnz = len(vals)

print(f"✔ Matriz lida: dimensão = {n} x {n}, nnz = {nnz}")

# Criar matriz PETSc
A = PETSc.Mat().create()
A.setSizes([n, n])
A.setType(PETSc.Mat.Type.AIJ)
A.setUp()

# Inserir valores
for r, c, v in zip(rows, cols, vals):
    A[r, c] = v

A.assemblyBegin()
A.assemblyEnd()

print("✔ Matriz PETSc montada!")


# ---------------------------------------------------------
# 2. Ler RHS binário (EX: rhs_10_1.bin)
# ---------------------------------------------------------

rhs_file = "rhs_10_1.bin"
print(f"Lendo RHS binário: {rhs_file}")

with open(rhs_file, "rb") as f:
    data = f.read()        # <-- ESTA LINHA ERA NECESSÁRIA

size_bytes = len(data)
print(f"📦 Tamanho do arquivo (bytes): {size_bytes}")

# Tentar deduzir formato
possible_float32 = (size_bytes % 4 == 0)
possible_float64 = (size_bytes % 8 == 0)

rhs_n32 = size_bytes // 4 if possible_float32 else -1
rhs_n64 = size_bytes // 8 if possible_float64 else -1

print("Possíveis formatos detectados:")
print(" • float32 →", rhs_n32)
print(" • float64 →", rhs_n64)

dtype = None
skip_first = False

# Preferimos float64
if rhs_n64 == n:
    dtype = np.float64
elif rhs_n64 == n + 1:
    dtype = np.float64
    skip_first = True
elif rhs_n32 == n:
    dtype = np.float32
elif rhs_n32 == n + 1:
    dtype = np.float32
    skip_first = True
else:
    raise RuntimeError("❌ Não foi possível determinar automaticamente o formato do RHS.")

rhs = np.frombuffer(data, dtype=dtype)

if skip_first:
    print("ℹ Detectado 1 valor extra no início (header). Ignorando a primeira entrada.")
    rhs = rhs[1:]

print(f"✔ RHS carregado! tamanho = {len(rhs)}, dtype = {rhs.dtype}")

# Criar vetor PETSc
b = PETSc.Vec().createSeq(n)
b.setArray(rhs)

# ---------------------------------------------------------
# 3. Resolver Ax = b com GMRES
# ---------------------------------------------------------

x = PETSc.Vec().createSeq(n)

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType("gmres")
ksp.getPC().setType("ilu")
ksp.setTolerances(rtol=1e-8, max_it=500)

print("🔄 Resolvendo sistema...")

ksp.solve(b, x)

reason = ksp.getConvergedReason()
its = ksp.getIterationNumber()
resnorm = ksp.getResidualNorm()

converged = (reason > 0)

print(f"✔ Convergiu? {converged}")
print(f"  Motivo da convergência (reason): {reason}")
print(f"  Iterações: {its}")
print(f"  ||residual|| = {resnorm}")

# Salvar solução
sol = np.array(x)
np.savetxt("solution.txt", sol)

print("✔ Solução salva em solution.txt")
