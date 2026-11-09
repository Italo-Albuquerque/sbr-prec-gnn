import pandas as pd

#Carrega os dados reais da Fase 1
df = pd.read_csv("ilu_all_results.csv")
df_ok = df[df["converged"] == True]

#Calcula o tempo médio para cada matriz e cada k (uniforme)
mean_times = df_ok.groupby(["matrix_id", "k"])["time"].mean().reset_index()

#Tempos locais e k locais
dados_locais = {
    10:  {"k": "(1,3)", "time": 0.000642}, 30:  {"k": "(2,3)", "time": 0.000681},
    45:  {"k": "(1,1)", "time": 0.000572}, 51:  {"k": "(3,2)", "time": 0.000675},
    54:  {"k": "(1,3)", "time": 0.000382}, 57:  {"k": "(1,2)", "time": 0.000529},
    61:  {"k": "(1,0)", "time": 0.000474}, 67:  {"k": "(3,3)", "time": 0.000472},
    68:  {"k": "(2,3)", "time": 0.000367}, 69:  {"k": "(3,2)", "time": 0.000479},
    76:  {"k": "(3,1)", "time": 0.000583}, 89:  {"k": "(2,3)", "time": 0.000592},
    96:  {"k": "(1,2)", "time": 0.000575}, 98:  {"k": "(1,0)", "time": 0.000549},
    99:  {"k": "(3,1)", "time": 0.000472}, 105: {"k": "(3,3)", "time": 0.000432}
}

print("Matriz & \(k\) uniforme & Tempo uniforme (s) & \(k\) locais & Tempo local (s) & Ganho (\\%) \\\\ \\midrule")
contagem = {0: 0, 1: 0, 2: 0, 3: 0}

for mat in sorted(mean_times["matrix_id"].unique()):
    row = mean_times[mean_times["matrix_id"] == mat]
    best_row = row.loc[row["time"].idxmin()]
    
    k_uni = int(best_row["k"])
    t_uni = best_row["time"]
    contagem[k_uni] += 1
    
    t_loc = dados_locais[mat]["time"]
    k_loc = dados_locais[mat]["k"]
    
    ganho = (t_uni / t_loc - 1) * 100
    
    # Formata com vírgula decimal e substitui o ponto
    t_uni_str = f"{t_uni:.6f}".replace('.', ',')
    t_loc_str = f"{t_loc:.6f}".replace('.', ',')
    ganho_str = f"{ganho:.2f}".replace('.', ',')
    
    print(f"{mat} & {k_uni} & {t_uni_str} & {k_loc} & {t_loc_str} & {ganho_str} \\\\")

print(f"\\midrule")
print(f"Contagem de vitorias: k=0 ({contagem[0]}), k=1 ({contagem[1]}), k=2 ({contagem[2]}), k=3 ({contagem[3]})")
