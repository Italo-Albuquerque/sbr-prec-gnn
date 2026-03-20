from petsc4py import PETSc
import numpy as np
import re

DUMP_FILE = "mat_10.txt"   # mude o nome aqui se quiser outro sistema

rows = []
cols = []
vals = []

gi = None
gj = None

# regex para pegar o valor do block = [ v 0 ]
block_re = re.compile(r"block\s*=\s*\[\s*([-+0-9.eE]+)")

with open(DUMP_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        if line.startswith("gi ="):
            gi = int(line.split("=")[1])
        elif line.startswith("gj ="):
            gj = int(line.split("=")[1])
        elif "block" in line:
            m = block_re.search(line)
            if m is None:
                continue
            v = float(m.group(1))
            rows.append(gi)
            cols.append(gj)
            vals.append(v)

rows = np.array(rows, dtype=int)
cols = np.array(cols, dtype=int)
vals = np.array(vals, dtype=float)

if rows.size == 0:
    raise RuntimeError("Não achei nenhuma entrada gi/gj/block no dump!")

n = int(max(rows.max(), cols.max()) + 1)

# Cria matriz PETSc em formato AIJ (CSR)
A = PETSc.Mat().create()
A.setSizes([n, n])
A.setType(PETSc.Mat.Type.AIJ)
A.setUp()

for i, j, v in zip(rows, cols, vals):
    A.setValue(int(i), int(j), float(v))

A.assemblyBegin()
A.assemblyEnd()

info = A.getInfo()
print("Matriz PETSc criada a partir do dump:")
print("  Tamanho:", A.getSize())
print("  NNZ:", int(info["nz_used"]))
