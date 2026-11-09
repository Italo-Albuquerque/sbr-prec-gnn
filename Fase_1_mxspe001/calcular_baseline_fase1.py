import pandas as pd

# Lê os resultados da Fase 1
df = pd.read_csv("ilu_all_results.csv")
df_ok = df[df["converged"] == True]

# Calcula o tempo médio para cada matriz e cada k
mean_times = df_ok.groupby(["matrix_id", "k"])["time"].mean().reset_index()

# Conta quantas vezes cada k foi o melhor (rótulos dos nós)
contagem = {0: 0, 1: 0, 2: 0, 3: 0}
for mat in mean_times["matrix_id"].unique():
    row = mean_times[mean_times["matrix_id"] == mat]
    # Pega os dois melhores (um para cada subdomínio)
    top2 = row.nsmallest(2, "time")["k"].values
    for k in top2:
        contagem[k] += 1

total_nos = sum(contagem.values())
maioria = max(contagem, key=contagem.get)
acuracia_maioria = (contagem[maioria] / total_nos) * 100

print(f"Distribuição dos nós: {contagem}")
print(f"Classe majoritária: k={maioria}")
print(f"Acurácia do classificador majoritário: {acuracia_maioria:.2f}%")
