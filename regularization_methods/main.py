import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ----------------------------
# 1. Генерация данных
# ----------------------------
np.random.seed(42)
n = 200

# Два сильно коррелированных признака
X1 = np.random.randn(n)
X2 = X1 + np.random.normal(scale=0.1, size=n)   # почти копия X1

# Ещё один независимый признак
X3 = np.random.randn(n)

# Целевая переменная
y = 3*X1 - 2*X2 + 0.5*X3 + np.random.normal(scale=0.5, size=n)

# Матрица признаков
X = np.vstack([X1, X2, X3]).T

# Разделение на train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------
# 2. Обучение моделей
# ----------------------------
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "Coefficients": model.coef_,
        "MSE": mean_squared_error(y_test, y_pred)
    }

# ----------------------------
# 3. Печать коэффициентов
# ----------------------------
print("=== Коэффициенты моделей ===")
coef_table = pd.DataFrame({
    name: res["Coefficients"] for name, res in results.items()
}, index=["X1", "X2", "X3"])
print(coef_table)

print("\n=== Ошибки моделей (MSE) ===")
for name, res in results.items():
    print(f"{name}: {res['MSE']:.4f}")

# ----------------------------
# 4. Визуализация предсказаний
# ----------------------------
plt.figure(figsize=(10, 6))

plt.plot(y_test, label="Истинные значения", marker="o")
plt.plot(models["Linear"].predict(X_test), label="Linear")
plt.plot(models["Ridge"].predict(X_test), label="Ridge")
plt.plot(models["Lasso"].predict(X_test), label="Lasso")
plt.plot(models["ElasticNet"].predict(X_test), label="ElasticNet")

plt.legend()
plt.title("Сравнение предсказаний моделей")
plt.xlabel("Наблюдение")
plt.ylabel("y")
plt.grid(True)
plt.tight_layout()

plt.savefig("regularization_demo.png", dpi=200)
plt.show()

print("\nГрафик сохранён как regularization_demo.png")

