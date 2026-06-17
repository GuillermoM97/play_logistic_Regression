"""Utilidades compartidas para el curso de regresión logística con Iris.

El módulo centraliza la carga del dataset binario, el split reproducible,
el guardado/carga de datos procesados y las visualizaciones recurrentes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42

COLOR_PALETTE = {
    "versicolor": "#0072B2",
    "virginica": "#D55E00",
    "decision": "#009E73",
    "neutral": "#4D4D4D",
    "grid": "#E5E7EB",
}


FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def set_plot_style() -> None:
    """Configura un estilo visual consistente para todas las figuras."""

    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette=[COLOR_PALETTE["versicolor"], COLOR_PALETTE["virginica"]],
    )
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": True,
            "grid.color": COLOR_PALETTE["grid"],
        }
    )


def load_binary_iris() -> tuple[pd.DataFrame, pd.Series]:
    """Carga Iris y conserva solo versicolor y virginica.

    La variable objetivo se relabela de forma pedagógica: versicolor=0 y
    virginica=1. Setosa se excluye porque es casi trivialmente separable y
    produciría una demostración menos informativa de la regresión logística.

    Devuelve:
        X: DataFrame con las cuatro variables originales de Iris.
        y: Serie binaria con virginica como clase positiva.
    """

    iris = load_iris()
    X_all = pd.DataFrame(iris.data, columns=FEATURE_NAMES)
    y_all = pd.Series(iris.target, name="species_original")

    mask = y_all.isin([1, 2])
    X = X_all.loc[mask].reset_index(drop=True)
    y = (y_all.loc[mask].reset_index(drop=True) == 2).astype(int)
    y.name = "virginica"
    return X, y


def get_train_test_split(
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    test_size: float = 0.30,
    random_state: int = RANDOM_STATE,
) -> dict[str, object]:
    """Crea un split estratificado y versiones escaladas/no escaladas.

    Parámetros:
        X: Matriz de variables. Si es None, se carga Iris binario.
        y: Variable objetivo. Si es None, se carga Iris binario.
        test_size: Proporción destinada al conjunto de prueba.
        random_state: Semilla para reproducibilidad.

    Devuelve:
        Diccionario con train/test sin escalar, train/test escalados,
        etiquetas, nombres de variables y el StandardScaler ajustado.
    """

    if X is None or y is None:
        X, y = load_binary_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled_array,
        columns=X.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled_array,
        columns=X.columns,
        index=X_test.index,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "feature_names": list(X.columns),
    }


def save_processed_splits(split_data: dict[str, object], output_dir: str | Path) -> Path:
    """Guarda los splits compartidos en formato NPZ pequeño y reproducible."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / "iris_binary_splits.npz"

    np.savez(
        file_path,
        X_train=split_data["X_train"].to_numpy(),
        X_test=split_data["X_test"].to_numpy(),
        X_train_scaled=split_data["X_train_scaled"].to_numpy(),
        X_test_scaled=split_data["X_test_scaled"].to_numpy(),
        y_train=split_data["y_train"].to_numpy(),
        y_test=split_data["y_test"].to_numpy(),
        feature_names=np.array(split_data["feature_names"]),
        scaler_mean=split_data["scaler"].mean_,
        scaler_scale=split_data["scaler"].scale_,
    )
    return file_path


def load_processed_splits(data_dir: str | Path) -> dict[str, object]:
    """Carga los splits generados por el notebook 00.

    Parámetros:
        data_dir: Carpeta `data/processed` donde vive `iris_binary_splits.npz`.

    Devuelve:
        Diccionario con DataFrames y Series listos para modelar.
    """

    file_path = Path(data_dir) / "iris_binary_splits.npz"
    if not file_path.exists():
        raise FileNotFoundError(
            "No se encontro iris_binary_splits.npz. Ejecuta primero "
            "notebooks/00_datos_y_eda.ipynb."
        )

    data = np.load(file_path, allow_pickle=False)
    feature_names = data["feature_names"].tolist()

    return {
        "X_train": pd.DataFrame(data["X_train"], columns=feature_names),
        "X_test": pd.DataFrame(data["X_test"], columns=feature_names),
        "X_train_scaled": pd.DataFrame(data["X_train_scaled"], columns=feature_names),
        "X_test_scaled": pd.DataFrame(data["X_test_scaled"], columns=feature_names),
        "y_train": pd.Series(data["y_train"].astype(int), name="virginica"),
        "y_test": pd.Series(data["y_test"].astype(int), name="virginica"),
        "feature_names": feature_names,
        "scaler_mean": data["scaler_mean"],
        "scaler_scale": data["scaler_scale"],
    }


def plot_decision_boundary(model, X_2d, y, feature_names, title):
    """Dibuja la frontera de decisión de un modelo entrenado con dos variables.

    Parámetros:
        model: Estimador con `predict_proba` o `predict`.
        X_2d: DataFrame/array con exactamente dos columnas.
        y: Etiquetas binarias alineadas con X_2d.
        feature_names: Nombres de las dos variables.
        title: Título descriptivo de la figura.

    Devuelve:
        Tupla `(fig, ax)` para permitir ajustes desde el notebook.
    """

    X_plot = pd.DataFrame(X_2d, columns=feature_names)
    y_plot = pd.Series(y).reset_index(drop=True)

    x_min, x_max = X_plot.iloc[:, 0].min() - 0.4, X_plot.iloc[:, 0].max() + 0.4
    y_min, y_max = X_plot.iloc[:, 1].min() - 0.4, X_plot.iloc[:, 1].max() + 0.4
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = pd.DataFrame(
        np.c_[xx.ravel(), yy.ravel()],
        columns=feature_names,
    )

    if hasattr(model, "predict_proba"):
        zz = model.predict_proba(grid)[:, 1]
    else:
        zz = model.predict(grid)
    zz = zz.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 5))
    contour = ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.25)
    ax.contour(
        xx,
        yy,
        zz,
        levels=[0.5],
        colors=[COLOR_PALETTE["decision"]],
        linewidths=2,
    )
    scatter = ax.scatter(
        X_plot.iloc[:, 0],
        X_plot.iloc[:, 1],
        c=y_plot,
        cmap="RdBu_r",
        edgecolor="white",
        linewidth=0.7,
        s=55,
    )
    ax.set_title(title)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend(
        handles=scatter.legend_elements()[0],
        labels=["versicolor (0)", "virginica (1)"],
        title="Clase",
        loc="best",
    )
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Probabilidad estimada de virginica")
    fig.tight_layout()
    return fig, ax


def plot_sigmoid():
    """Dibuja la función logística y señala el umbral probabilístico 0.5."""

    z = np.linspace(-8, 8, 500)
    probability = 1 / (1 + np.exp(-z))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z, probability, color=COLOR_PALETTE["decision"], linewidth=2.5)
    ax.axhline(0.5, color=COLOR_PALETTE["neutral"], linestyle="--", linewidth=1.2)
    ax.axvline(0, color=COLOR_PALETTE["neutral"], linestyle="--", linewidth=1.2)
    ax.annotate(
        "log-odds = 0\nprobabilidad = 0.5",
        xy=(0, 0.5),
        xytext=(1.0, 0.25),
        arrowprops={"arrowstyle": "->", "color": COLOR_PALETTE["neutral"]},
    )
    ax.set_title("Función logística: del log-odds a la probabilidad")
    ax.set_xlabel("Predictor lineal / log-odds")
    ax.set_ylabel("Probabilidad P(y=1 | x)")
    ax.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    return fig, ax


def plot_coefficients(coefs, feature_names, title):
    """Genera una gráfica de coeficientes de un modelo logístico en barras ordenadas."""

    coef_df = pd.DataFrame({"feature": feature_names, "coeficiente": np.ravel(coefs)})
    coef_df = coef_df.reindex(coef_df["coeficiente"].abs().sort_values().index)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = np.where(
        coef_df["coeficiente"] >= 0,
        COLOR_PALETTE["virginica"],
        COLOR_PALETTE["versicolor"],
    )
    ax.barh(coef_df["feature"], coef_df["coeficiente"], color=colors)
    ax.axvline(0, color=COLOR_PALETTE["neutral"], linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Coeficiente en escala log-odds")
    ax.set_ylabel("Variable")
    fig.tight_layout()
    return fig, ax


def plot_regularization_path(C_values, coefs_matrix, feature_names, title):
    """Dibuja la ruta de coeficientes contra C en escala logarítmica.

    En scikit-learn, C controla de forma inversa la fuerza de regularización:
    valores pequeños implican penalización fuerte y valores grandes implican
    penalización débil.
    """

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for feature_idx, feature_name in enumerate(feature_names):
        ax.plot(
            C_values,
            coefs_matrix[:, feature_idx],
            marker="o",
            linewidth=1.8,
            label=feature_name,
        )
    ax.axhline(0, color=COLOR_PALETTE["neutral"], linewidth=1)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("C (mayor C = menor penalización)")
    ax.set_ylabel("Coeficiente estandarizado")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig, ax
