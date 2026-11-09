cd ~/sbr-prec-gnn/data/spe10_conjunto_A

OUT_CSV="resultado_k012_conjuntoA.csv"
LOG="log_baseline.txt"

echo "matrix_id,rhs,k,n,its,residual,reason,time,status" > $OUT_CSV

# IDs das 16 matrizes
for mat_id in 10 30 45 51 54 57 61 67 68 69 76 89 96 98 99 105; do
    echo "Processando mat_$mat_id" | tee -a $LOG

    # Pega os 3 primeiros RHS
    rhs_list=$(ls rhs/rhs_${mat_id}_*.petscbin 2>/dev/null | sort | head -3)

    if [ -z "$rhs_list" ]; then
        echo "  Nenhum RHS encontrado para mat_$mat_id" | tee -a $LOG
        continue
    fi

    # Loop para rodar cada um dos 3 RHS
    for rhs_path in $rhs_list; do
        rhs_name=$(basename $rhs_path)
        echo "  RHS: $rhs_name" | tee -a $LOG
        mpiexec -n 4 --oversubscribe python3 test_k012_spe10.py $mat_id $rhs_name 2>&1 | grep -E "^[0-9]+," >> $OUT_CSV
    done
done

echo "Concluido. Resultados em $OUT_CSV"
