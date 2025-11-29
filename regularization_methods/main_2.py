import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

RANDOM_STATE = 42


def get_models():
    """
    Набор моделей, одинаково используемый во всех экспериментах.
    Параметры подобраны так, чтобы:
    - при сильной мультиколлинеарности выигрывал Ridge;
    - при разрежной high-dimensional модели выигрывал Lasso.
    """
    return {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=10_000),
        "ElasticNet": ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=RANDOM_STATE,
            max_iter=10_000,
        ),
    }


def run_and_report(name: str,
                   X: np.ndarray,
                   y: np.ndarray,
                   feature_names=None,
                   plot_filename: str | None = None):
    """
    Общая функция:
    - делит данные на train/test;
    - обучает все модели;
    - печатает таблицу коэффициентов, MSE и число ненулевых коэффициентов;
    - по желанию рисует график предсказаний.
    """
    print("=" * 80)
    print(name)
    print("=" * 80)

    if feature_names is None:
        feature_names = [f"x{i + 1}" for i in range(X.shape[1])]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )

    models = get_models()
    coefs: dict[str, np.ndarray] = {}
    intercepts: dict[str, float] = {}
    mses: dict[str, float] = {}
    non_zero: dict[str, int] = {}
    y_pred_dict: dict[str, np.ndarray] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_pred_dict[model_name] = y_pred
        mses[model_name] = mean_squared_error(y_test, y_pred)
        coefs[model_name] = model.coef_
        intercepts[model_name] = float(model.intercept_)
        non_zero[model_name] = int(np.sum(np.abs(model.coef_) > 1e-4))

    # Таблица коэффициентов
    coef_table = pd.DataFrame(coefs, index=feature_names)
    print("\nКоэффициенты (без свободного члена):")
    # округление для удобства чтения
    print(coef_table.round(3))

    # Свободные члены
    print("\nСвободные члены (intercept):")
    print(pd.Series(intercepts).round(3))

    # MSE
    print("\nMSE на тестовой выборке:")
    mse_series = pd.Series(mses).sort_values()
    print(mse_series.round(4))

    # Число ненулевых коэффициентов
    print("\nЧисло ненулевых коэффициентов (sparsity):")
    print(pd.Series(non_zero))

    # График предсказаний
    if plot_filename is not None:
        order = np.argsort(y_test)  # для красивого графика сортируем по истинным y

        plt.figure(figsize=(8, 5))
        plt.plot(y_test[order], label="Истинные значения", linewidth=2)

        for model_name, y_pred in y_pred_dict.items():
            plt.plot(y_pred[order], label=model_name, alpha=0.9)

        plt.title(name)
        plt.xlabel("Наблюдения (отсортированы по истинному y)")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(plot_filename, dpi=200)
        plt.show()

    print()  # пустая строка между экспериментами

    return mse_series


def experiment_multicollinear():
    """
    Эксперимент 1: сильная мультиколлинеарность (X1 ~ X2), 3 признака.
    В этом сценарии Ridge даёт наименьший MSE.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    n_samples = 200

    # X1 — базовый признак
    X1 = rng.normal(0, 1, size=n_samples)
    # X2 ≈ X1 (сильная мультиколлинеарность)
    X2 = X1 + rng.normal(0, 0.01, size=n_samples)
    # X3 — независимый
    X3 = rng.normal(0, 1, size=n_samples)

    X = np.vstack([X1, X2, X3]).T
    feature_names = ["X1", "X2", "X3"]

    # Истинные коэффициенты
    true_beta = np.array([3.0, -2.0, 0.5])
    y = X @ true_beta + rng.normal(0, 1.0, size=n_samples)

    print("Истинные коэффициенты (мультиколлинеарность):")
    print(pd.Series(true_beta, index=feature_names).round(3))

    run_and_report(
        name="Эксперимент 1: сильная мультиколлинеарность (Ridge выигрывает)",
        X=X,
        y=y,
        feature_names=feature_names,
        plot_filename="regularization_multicollinear.png",
    )


def experiment_sparse_highdim():
    """
    Эксперимент 2: разрежная high-dimensional модель.
    - 40 признаков, реально важны только первые 5.
    - В таком сценарии Lasso обычно даёт наименьший MSE и мало ненулевых весов.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    n_samples = 40
    n_features = 40
    n_informative = 5

    X = rng.normal(size=(n_samples, n_features))

    beta = np.zeros(n_features)
    beta[:n_informative] = (
        rng.uniform(0.5, 2.0, size=n_informative)
        * rng.choice([-1, 1], size=n_informative)
    )

    y = X @ beta + rng.normal(0, 1.0, size=n_samples)

    feature_names = [f"x{i + 1}" for i in range(n_features)]

    print("Истинные коэффициенты (разрежная модель, первые 10):")
    print(pd.Series(beta[:10], index=feature_names[:10]).round(3))

    run_and_report(
        name="Эксперимент 2: разрежная high-dimensional модель (Lasso выигрывает)",
        X=X,
        y=y,
        feature_names=feature_names,
        plot_filename="regularization_sparse.png",
    )


if __name__ == "__main__":
    experiment_multicollinear()
    experiment_sparse_highdim()

