import sys
import re
import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import mmwrite

if len(sys.argv) != 3:
    print("Uso: python3 dump_to_mtx.py mat_10.txt mat_10.mtx")
    sys.exit(1)

INPUT = sys.argv[1]
OUTPUT = sys.argv[2]

rows = []
cols = []
vals = []

gi = None
gj = None

# regex para extrair o valor dentro do block
block_re = re.compile(r"block\s*=\s*\[\s*([-+0-9.eE]+)")

with open(INPUT, "r") as f:
    for line in f:
        line = line.strip()

        if line.startswith("gi ="):
            gi = int(line.split("=")[1])
        
        elif line.startswith("gj ="):
            gj = int(line.split("=")[1])

        elif "block" in line:
            m = block_re.search(line)
            if m:
                val = float(m.group(1))
                rows.append(gi)
                cols.append(gj)
                vals.append(val)

# monta a matriz esparsa
n = max(max(rows), max(cols)) + 1
A = coo_matrix((vals, (rows, cols)), shape=(n, n))

mmwrite(OUTPUT, A)
print(f"Gerado arquivo Matrix Market: {OUTPUT}")
print(f"Tamanho: {n} x {n}")
print(f"NNZ = {A.nnz}")
