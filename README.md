# Regresión logística con Iris: de la estadística clásica al machine learning moderno

Este repositorio es un curso práctico y autocontenido sobre regresión logística. La idea es estudiar el modelo desde sus fundamentos estadísticos, implementarlo desde cero, conectarlo con los Modelos Lineales Generalizados y avanzar hacia regularización, selección de hiperparámetros, aprendizaje sensible al costo y clases desbalanceadas.

El dataset usado en todos los notebooks es Iris en versión binaria: **versicolor vs virginica**. Se excluye setosa porque es linealmente separable del resto, lo que puede producir separación perfecta, coeficientes de máxima verosimilitud divergentes y métricas demasiado fáciles. Versicolor y virginica se solapan parcialmente, de modo que el problema conserva fronteras de decisión interesantes, coeficientes estables y métricas realistas. La clase positiva es siempre **virginica = 1** y la clase negativa es **versicolor = 0**.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
jupyter notebook
```

Ejecuta primero `notebooks/00_datos_y_eda.ipynb`, porque genera los splits compartidos en `data/processed/`. Los demás notebooks cargan esos mismos archivos para que las comparaciones sean reproducibles.

## Tabla de contenidos

| Notebook | Tema | Referencias clave |
|---|---|---|
| `00_datos_y_eda.ipynb` | Preparación del Iris binario, EDA, correlaciones, split y escalado. | Hosmer, Lemeshow & Sturdivant (2013) |
| `01_regresion_logistica_desde_cero.ipynb` | Logit, odds, MLE, descenso de gradiente y frontera de decisión. | Berkson (1944), Cox (1958), Agresti (2013), Bishop (2006) |
| `02_logistica_como_glm.ipynb` | Regresión logística como GLM, inferencia con statsmodels, deviance e IRLS. | Nelder & Wedderburn (1972), McCullagh & Nelder (1989), Agresti (2013) |
| `03_evaluacion_y_bondad_de_ajuste.ipynb` | Matriz de confusión, ROC, PR, calibración, Hosmer-Lemeshow y umbrales. | Hosmer & Lemeshow (1980), Hosmer et al. (2013), He & Garcia (2009) |
| `04_regularizacion_L2_ridge.ipynb` | Penalti L2, colinealidad, rutas de coeficientes y sesgo-varianza. | Hoerl & Kennard (1970), Hastie et al. (2009), Murphy (2012) |
| `05_regularizacion_L1_lasso.ipynb` | Penalti L1, sparsity, selección automática de variables y fronteras. | Tibshirani (1996), Hastie et al. (2009) |
| `06_elastic_net.ipynb` | Mezcla L1/L2, efecto de agrupamiento y solver SAGA. | Zou & Hastie (2005), Defazio et al. (2014), Murphy (2012) |
| `07_rutas_de_regularizacion_y_tuning.ipynb` | Validación cruzada, búsqueda de hiperparámetros y comparación de penaltis. | Friedman et al. (2010), Hastie et al. (2009), Pedregosa et al. (2011) |
| `08_cost_sensitive_y_desbalance.ipynb` | Desbalance artificial, pesos de clase, umbrales y SMOTE. | Elkan (2001), Chawla et al. (2002), He & Garcia (2009) |

## Mapa conceptual

El recorrido sigue tres bloques:

1. **Fundamentos estadísticos (NB01-NB03):** logit, máxima verosimilitud, GLM, inferencia, evaluación, calibración y bondad de ajuste.
2. **Regularización (NB04-NB07):** Ridge, Lasso, Elastic Net, rutas de regularización y selección rigurosa de hiperparámetros.
3. **Desbalance y costo (NB08):** decisiones con costos asimétricos, métricas adecuadas, ponderación de clases, ajuste de umbral y SMOTE.

La bibliografía completa está en [REFERENCIAS.md](REFERENCIAS.md).
