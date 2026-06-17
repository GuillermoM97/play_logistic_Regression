# Regresión logística con Iris y Titanic: de la estadística clásica al machine learning moderno

Este repositorio es un curso práctico y autocontenido sobre regresión logística. La idea es estudiar el modelo desde sus fundamentos estadísticos, implementarlo desde cero, conectarlo con los Modelos Lineales Generalizados y avanzar hacia regularización, selección de hiperparámetros, aprendizaje sensible al costo y clases desbalanceadas.

El repositorio tiene dos pistas equivalentes:

1. **Pista Iris (`notebooks/`)**: usa Iris en versión binaria, **versicolor = 0** vs **virginica = 1**. Se excluye setosa porque es linealmente separable del resto, lo que puede producir separación perfecta, coeficientes de máxima verosimilitud divergentes y métricas demasiado fáciles. Versicolor y virginica se solapan parcialmente, de modo que el problema conserva fronteras de decisión interesantes, coeficientes estables y métricas realistas.
2. **Pista Titanic (`notebooks_2/`)**: usa el dataset `titanic` de seaborn con objetivo binario **fallecido = 0** vs **sobreviviente = 1**. Esta pista añade faltantes, variables categóricas, codificación one-hot, riesgo de fuga de información y desbalance moderado.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
jupyter notebook
```

Para Iris, ejecuta primero `notebooks/00_datos_y_eda.ipynb`, porque genera `data/processed/iris_binary_splits.npz`.

Para Titanic, ejecuta primero `notebooks_2/00_datos_y_eda.ipynb`, porque genera `data/processed/titanic_binary_splits.npz`. El CSV oficial de seaborn-data queda en caché local en `data/raw/seaborn/titanic.csv` para que los notebooks funcionen de forma reproducible sin depender de una descarga durante clase.

## Tabla de contenidos: pista Iris

| Notebook | Tema | Referencias clave |
|---|---|---|
| `notebooks/00_datos_y_eda.ipynb` | Preparación del Iris binario, EDA, correlaciones, split y escalado. | Hosmer, Lemeshow & Sturdivant (2013) |
| `notebooks/01_regresion_logistica_desde_cero.ipynb` | Logit, odds, MLE, descenso de gradiente y frontera de decisión. | Berkson (1944), Cox (1958), Agresti (2013), Bishop (2006) |
| `notebooks/02_logistica_como_glm.ipynb` | Regresión logística como GLM, inferencia con statsmodels, deviance e IRLS. | Nelder & Wedderburn (1972), McCullagh & Nelder (1989), Agresti (2013) |
| `notebooks/03_evaluacion_y_bondad_de_ajuste.ipynb` | Matriz de confusión, ROC, PR, calibración, Hosmer-Lemeshow y umbrales. | Hosmer & Lemeshow (1980), Hosmer et al. (2013), He & Garcia (2009) |
| `notebooks/04_regularizacion_L2_ridge.ipynb` | Penalti L2, colinealidad, rutas de coeficientes y sesgo-varianza. | Hoerl & Kennard (1970), Hastie et al. (2009), Murphy (2012) |
| `notebooks/05_regularizacion_L1_lasso.ipynb` | Penalti L1, sparsity, selección automática de variables y fronteras. | Tibshirani (1996), Hastie et al. (2009) |
| `notebooks/06_elastic_net.ipynb` | Mezcla L1/L2, efecto de agrupamiento y solver SAGA. | Zou & Hastie (2005), Defazio et al. (2014), Murphy (2012) |
| `notebooks/07_rutas_de_regularizacion_y_tuning.ipynb` | Validación cruzada, búsqueda de hiperparámetros y comparación de penaltis. | Friedman et al. (2010), Hastie et al. (2009), Pedregosa et al. (2011) |
| `notebooks/08_cost_sensitive_y_desbalance.ipynb` | Desbalance artificial, pesos de clase, umbrales y SMOTE. | Elkan (2001), Chawla et al. (2002), He & Garcia (2009) |

## Tabla de contenidos: pista Titanic

| Notebook | Tema | Referencias clave |
|---|---|---|
| `notebooks_2/00_datos_y_eda.ipynb` | Preparación de Titanic binario, auditoría de fuga de información, EDA, split estratificado, imputación, one-hot encoding y escalado. | Hosmer, Lemeshow & Sturdivant (2013), Waskom (2021) |
| `notebooks_2/01_regresion_logistica_desde_cero.ipynb` | Logit, odds, MLE, descenso de gradiente y comparación con scikit-learn. | Berkson (1944), Cox (1958), Agresti (2013), Bishop (2006) |
| `notebooks_2/02_logistica_como_glm.ipynb` | Regresión logística como GLM binomial, odds ratios, deviance e inferencia. | Nelder & Wedderburn (1972), McCullagh & Nelder (1989), Agresti (2013) |
| `notebooks_2/03_evaluacion_y_bondad_de_ajuste.ipynb` | Matriz de confusión, ROC, PR, calibración, Hosmer-Lemeshow y umbrales. | Hosmer & Lemeshow (1980), Hosmer et al. (2013), He & Garcia (2009) |
| `notebooks_2/04_regularizacion_L2_ridge.ipynb` | Penalti L2, colinealidad, rutas de coeficientes y fronteras con regularización fuerte/débil. | Hoerl & Kennard (1970), Hastie et al. (2009), Murphy (2012) |
| `notebooks_2/05_regularizacion_L1_lasso.ipynb` | Penalti L1, sparsity, variables activas y frontera L1. | Tibshirani (1996), Hastie et al. (2009) |
| `notebooks_2/06_elastic_net.ipynb` | Mezcla L1/L2, `l1_ratio`, efecto de agrupamiento y solver SAGA. | Zou & Hastie (2005), Defazio et al. (2014), Murphy (2012) |
| `notebooks_2/07_rutas_de_regularizacion_y_tuning.ipynb` | Validación cruzada, búsqueda de `C` y `l1_ratio`, comparación final de penaltis. | Friedman et al. (2010), Hastie et al. (2009), Pedregosa et al. (2011) |
| `notebooks_2/08_cost_sensitive_y_desbalance.ipynb` | Desbalance artificial, `class_weight`, umbral sensible al costo y SMOTE. | Elkan (2001), Chawla et al. (2002), He & Garcia (2009) |

## Mapa conceptual

El recorrido sigue tres bloques en ambas pistas:

1. **Fundamentos estadísticos (NB01-NB03):** logit, máxima verosimilitud, GLM, inferencia, evaluación, calibración y bondad de ajuste.
2. **Regularización (NB04-NB07):** Ridge, Lasso, Elastic Net, rutas de regularización y selección rigurosa de hiperparámetros.
3. **Desbalance y costo (NB08):** decisiones con costos asimétricos, métricas adecuadas, ponderación de clases, ajuste de umbral y SMOTE.

La bibliografía completa está en [REFERENCIAS.md](REFERENCIAS.md).
