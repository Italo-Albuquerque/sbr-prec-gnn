import pandas as pd
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("teacher_soft_labels.csv")
y_q1 = df["q_g1"].values
y_q2 = df["q_g2"].values

mae_trivial_q1 = mean_absolute_error(y_q1, [y_q1.mean()] * len(y_q1))
mae_trivial_q2 = mean_absolute_error(y_q2, [y_q2.mean()] * len(y_q2))

print(f"MAE Trivial (média) q1: {mae_trivial_q1:.4f}")
print(f"MAE Trivial (média) q2: {mae_trivial_q2:.4f}")
