import numpy as np
from scipy.io import mmwrite
from scipy.sparse import coo_matrix
import sys

if len(sys.argv) < 3:
    print("Uso: python txt_to_mtx.py input.csv output.mtx")
    sys.exit(1)

INPUT = sys.argv[1]
OUTPUT = sys.argv[2]

print(f"Lendo arquivo CSV: {INPUT}")

try:
    data = np.loadtxt(INPUT, delimiter=';', skiprows=1)
except Exception as e:
    print("Erro ao ler o CSV:", e)
    sys.exit(1)

# Se não há dados após o cabeçalho:
if data.size == 0:
    print("Este CSV não contém dados numéricos de matriz, só cabeçalho/metadados.")
    sys.exit(1)

# Garante formato 2D
if data.ndim == 1:
    data = data.reshape(1, -1)

rows = data[:, 0].astype(int)
cols = data[:, 1].astype(int)
vals = data[:, 2]

# Se os índices começam em 1 (1-based), converte para 0-based:
if rows.min() == 1 or cols.min() == 1:
    rows -= 1
    cols -= 1

n = max(rows.max(), cols.max()) + 1
A = coo_matrix((vals, (rows, cols)), shape=(n, n))

print(f"Matriz carregada. n={n}, nnz={A.nnz}")
print(f"Salvando Matrix Market em: {OUTPUT}")
mmwrite(OUTPUT, A)
print("Arquivo .mtx gerado com sucesso!")
