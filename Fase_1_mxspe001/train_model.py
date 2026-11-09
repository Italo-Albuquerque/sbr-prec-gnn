import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("dataset.csv")

X = df[["n", "nnz", "density"]]
y = df["best_k"]

# só treina se houver exemplos suficientes
if len(df) < 5:
    print("Poucos exemplos para treino.")
    print(df)
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\nAcurácia =", acc)

print("\nImportância das features:")
for name, imp in zip(X.columns, model.feature_importances_):
    print(f"{name}: {imp:.4f}")

print("\nPredições:")
for real, pred in zip(y_test, y_pred):
    print(f"real={real}  pred={pred}")
