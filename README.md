# Regresion logistica con Iris: de la estadistica clasica al machine learning moderno

Este repositorio es un curso practico y autocontenido sobre regresion logistica. La idea es estudiar el modelo desde sus fundamentos estadisticos, implementarlo desde cero, conectarlo con los Modelos Lineales Generalizados y avanzar hacia regularizacion, seleccion de hiperparametros, aprendizaje sensible al costo y clases desbalanceadas.

El dataset usado en todos los notebooks es Iris en version binaria: **versicolor vs virginica**. Se excluye setosa porque es linealmente separable del resto, lo que puede producir separacion perfecta, coeficientes de maxima verosimilitud divergentes y metricas demasiado faciles. Versicolor y virginica se solapan parcialmente, de modo que el problema conserva fronteras de decision interesantes, coeficientes estables y metricas realistas. La clase positiva es siempre **virginica = 1** y la clase negativa es **versicolor = 0**.

## Instalacion

```bash
pip install -r requirements.txt
```

## Ejecucion

```bash
jupyter notebook
```

Ejecuta primero `notebooks/00_datos_y_eda.ipynb`, porque genera los splits compartidos en `data/processed/`. Los demas notebooks cargan esos mismos archivos para que las comparaciones sean reproducibles.

## Tabla de contenidos

| Notebook | Tema | Referencias clave |
|---|---|---|
| `00_datos_y_eda.ipynb` | Preparacion del Iris binario, EDA, correlaciones, split y escalado. | Hosmer, Lemeshow & Sturdivant (2013) |
| `01_regresion_logistica_desde_cero.ipynb` | Logit, odds, MLE, descenso de gradiente y frontera de decision. | Berkson (1944), Cox (1958), Agresti (2013), Bishop (2006) |
| `02_logistica_como_glm.ipynb` | Regresion logistica como GLM, inferencia con statsmodels, deviance e IRLS. | Nelder & Wedderburn (1972), McCullagh & Nelder (1989), Agresti (2013) |
| `03_evaluacion_y_bondad_de_ajuste.ipynb` | Matriz de confusion, ROC, PR, calibracion, Hosmer-Lemeshow y umbrales. | Hosmer & Lemeshow (1980), Hosmer et al. (2013), He & Garcia (2009) |
| `04_regularizacion_L2_ridge.ipynb` | Penalti L2, colinealidad, rutas de coeficientes y sesgo-varianza. | Hoerl & Kennard (1970), Hastie et al. (2009), Murphy (2012) |
| `05_regularizacion_L1_lasso.ipynb` | Penalti L1, sparsity, seleccion automatica de variables y fronteras. | Tibshirani (1996), Hastie et al. (2009) |
| `06_elastic_net.ipynb` | Mezcla L1/L2, efecto de agrupamiento y solver SAGA. | Zou & Hastie (2005), Defazio et al. (2014), Murphy (2012) |
| `07_rutas_de_regularizacion_y_tuning.ipynb` | Validacion cruzada, busqueda de hiperparametros y comparacion de penaltis. | Friedman et al. (2010), Hastie et al. (2009), Pedregosa et al. (2011) |
| `08_cost_sensitive_y_desbalance.ipynb` | Desbalance artificial, pesos de clase, umbrales y SMOTE. | Elkan (2001), Chawla et al. (2002), He & Garcia (2009) |

## Mapa conceptual

El recorrido sigue tres bloques:

1. **Fundamentos estadisticos (NB01-NB03):** logit, maxima verosimilitud, GLM, inferencia, evaluacion, calibracion y bondad de ajuste.
2. **Regularizacion (NB04-NB07):** Ridge, Lasso, Elastic Net, rutas de regularizacion y seleccion rigurosa de hiperparametros.
3. **Desbalance y costo (NB08):** decisiones con costos asimetricos, metricas adecuadas, ponderacion de clases, ajuste de umbral y SMOTE.

La bibliografia completa esta en [REFERENCIAS.md](REFERENCIAS.md).
