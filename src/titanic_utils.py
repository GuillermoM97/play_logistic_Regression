"""Utilidades compartidas para la pista Titanic de regresión logística.

El módulo carga el dataset Titanic de seaborn desde una caché local del repo,
prepara un problema binario reproducible y centraliza las visualizaciones que
se repiten en los notebooks de `notebooks_2`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEABORN_DATA_HOME = PROJECT_ROOT / "data" / "raw" / "seaborn"
PROCESSED_FILE_NAME = "titanic_binary_splits.npz"

TARGET_NAME = "survived"
POSITIVE_CLASS_LABEL = "sobreviviente"
NEGATIVE_CLASS_LABEL = "fallecido"

NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked", "alone"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LEAKAGE_COLUMNS = ["alive"]
DUPLICATE_OR_DERIVED_COLUMNS = ["class", "who", "adult_male", "deck", "embark_town"]

COLOR_PALETTE = {
    "died": "#0072B2",
    "survived": "#D55E00",
    "decision": "#009E73",
    "neutral": "#4D4D4D",
    "grid": "#E5E7EB",
}


def set_plot_style() -> None:
    """Configura un estilo visual consistente para todas las figuras."""

    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette=[COLOR_PALETTE["died"], COLOR_PALETTE["survived"]],
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


def load_titanic_raw() -> pd.DataFrame:
    """Carga Titanic desde la caché local de seaborn-data.

    El archivo `data/raw/seaborn/titanic.csv` es la versión usada por
    `seaborn.load_dataset("titanic")`. Pasar `data_home` evita que cada
    ejecución intente escribir en la caché global del usuario.
    """

    SEABORN_DATA_HOME.mkdir(parents=True, exist_ok=True)
    return sns.load_dataset("titanic", data_home=str(SEABORN_DATA_HOME))


def load_titanic_binary() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepara el problema binario de supervivencia del Titanic.

    La clase positiva es `survived = 1`, es decir, pasajero sobreviviente. Se
    excluyen columnas que duplican el objetivo (`alive`) o variables derivadas/
    redundantes que complicarían la interpretación pedagógica inicial.

    Devuelve:
        X: DataFrame con variables predictoras crudas seleccionadas.
        y: Serie binaria con `survived` como clase positiva.
        df: DataFrame original de seaborn para EDA y auditoría.
    """

    df = load_titanic_raw()
    X = df.loc[:, FEATURE_COLUMNS].copy()
    y = df[TARGET_NAME].astype(int).rename(TARGET_NAME)
    return X, y, df


def make_titanic_preprocessor() -> ColumnTransformer:
    """Construye el preprocesador usado en todos los modelos.

    - Variables numéricas: imputación por mediana y estandarización.
    - Variables categóricas: imputación por moda y one-hot encoding con una
      categoría de referencia (`drop="first"`) para reducir colinealidad.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Devuelve nombres legibles después del ColumnTransformer."""

    return [str(name) for name in preprocessor.get_feature_names_out()]


def get_train_test_split(
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    test_size: float = 0.30,
    random_state: int = RANDOM_STATE,
) -> dict[str, object]:
    """Crea un split estratificado y versiones crudas/procesadas.

    Parámetros:
        X: Variables predictoras crudas. Si es None, se carga Titanic.
        y: Variable objetivo binaria. Si es None, se carga Titanic.
        test_size: Proporción destinada al conjunto de prueba.
        random_state: Semilla para reproducibilidad.

    Devuelve:
        Diccionario con train/test crudos, train/test procesados,
        etiquetas, nombres de variables y el preprocesador ajustado.
    """

    if X is None or y is None:
        X, y, _ = load_titanic_binary()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = make_titanic_preprocessor()
    X_train_processed_array = preprocessor.fit_transform(X_train)
    X_test_processed_array = preprocessor.transform(X_test)
    feature_names = _feature_names(preprocessor)

    X_train_processed = pd.DataFrame(
        X_train_processed_array,
        columns=feature_names,
        index=X_train.index,
    )
    X_test_processed = pd.DataFrame(
        X_test_processed_array,
        columns=feature_names,
        index=X_test.index,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }


def save_processed_splits(split_data: dict[str, object], output_dir: str | Path) -> Path:
    """Guarda los splits compartidos en formato NPZ pequeño y reproducible."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / PROCESSED_FILE_NAME

    np.savez(
        file_path,
        X_train_processed=split_data["X_train_processed"].to_numpy(),
        X_test_processed=split_data["X_test_processed"].to_numpy(),
        y_train=split_data["y_train"].to_numpy(),
        y_test=split_data["y_test"].to_numpy(),
        feature_names=np.array(split_data["feature_names"]),
        raw_feature_columns=np.array(FEATURE_COLUMNS),
        numeric_features=np.array(NUMERIC_FEATURES),
        categorical_features=np.array(CATEGORICAL_FEATURES),
        positive_class=np.array([POSITIVE_CLASS_LABEL]),
        negative_class=np.array([NEGATIVE_CLASS_LABEL]),
    )
    return file_path


def load_processed_splits(data_dir: str | Path) -> dict[str, object]:
    """Carga los splits generados por el notebook 00 de la pista Titanic."""

    file_path = Path(data_dir) / PROCESSED_FILE_NAME
    if not file_path.exists():
        raise FileNotFoundError(
            "No se encontró titanic_binary_splits.npz. Ejecuta primero "
            "notebooks_2/00_datos_y_eda.ipynb."
        )

    data = np.load(file_path, allow_pickle=False)
    feature_names = data["feature_names"].tolist()

    return {
        "X_train_processed": pd.DataFrame(
            data["X_train_processed"],
            columns=feature_names,
        ),
        "X_test_processed": pd.DataFrame(
            data["X_test_processed"],
            columns=feature_names,
        ),
        "y_train": pd.Series(data["y_train"].astype(int), name=TARGET_NAME),
        "y_test": pd.Series(data["y_test"].astype(int), name=TARGET_NAME),
        "feature_names": feature_names,
        "raw_feature_columns": data["raw_feature_columns"].tolist(),
        "numeric_features": data["numeric_features"].tolist(),
        "categorical_features": data["categorical_features"].tolist(),
    }


def plot_decision_boundary(model, X_2d, y, feature_names, title):
    """Dibuja la frontera de decisión de un modelo entrenado con dos variables."""

    X_plot = pd.DataFrame(X_2d, columns=feature_names)
    y_plot = pd.Series(y).reset_index(drop=True)

    x_padding = 0.5
    y_padding = 0.5
    x_min = X_plot.iloc[:, 0].min() - x_padding
    x_max = X_plot.iloc[:, 0].max() + x_padding
    y_min = X_plot.iloc[:, 1].min() - y_padding
    y_max = X_plot.iloc[:, 1].max() + y_padding
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
        s=45,
        alpha=0.9,
    )
    ax.set_title(title)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend(
        handles=scatter.legend_elements()[0],
        labels=["fallecido (0)", "sobreviviente (1)"],
        title="Clase",
        loc="best",
    )
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Probabilidad estimada de supervivencia")
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
    """Genera una gráfica de coeficientes logísticos en barras ordenadas."""

    coef_df = pd.DataFrame({"feature": feature_names, "coeficiente": np.ravel(coefs)})
    coef_df = coef_df.reindex(coef_df["coeficiente"].abs().sort_values().index)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = np.where(
        coef_df["coeficiente"] >= 0,
        COLOR_PALETTE["survived"],
        COLOR_PALETTE["died"],
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
