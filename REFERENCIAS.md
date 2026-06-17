# Referencias bibliograficas

## Fundamentos estadisticos

- Berkson, J. (1944). Application of the Logistic Function to Bio-Assay. *Journal of the American Statistical Association*, 39(227), 357-365. Aporta el termino logit y la funcion logistica como herramienta para modelar respuestas binarias. Usado en NB01.
- Cox, D. R. (1958). The Regression Analysis of Binary Sequences. *Journal of the Royal Statistical Society, Series B*, 20(2), 215-242. Formaliza la regresion para secuencias binarias y da la base estadistica moderna de la regresion logistica. Usado en NB01.
- Nelder, J. A., & Wedderburn, R. W. M. (1972). Generalized Linear Models. *Journal of the Royal Statistical Society, Series A*, 135(3), 370-384. Introduce el marco GLM que contiene a la regresion logistica como caso binomial con enlace logit. Usado en NB02.
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC. Desarrolla deviance, residuos, inferencia e IRLS para GLM. Usado en NB02.
- Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley. Explica modelos para datos categoricos, log-odds, odds ratios e inferencia para regresion logistica. Usado en NB01 y NB02.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley. Referencia aplicada para preparacion de datos, diagnostico, interpretacion y evaluacion de modelos logisticos. Usado en NB00 y NB03.
- Hosmer, D. W., & Lemeshow, S. (1980). Goodness of fit tests for the multiple logistic regression model. *Communications in Statistics - Theory and Methods*, 9(10), 1043-1069. Propone el test de bondad de ajuste implementado desde cero en el curso. Usado en NB03.

## Machine learning y regularizacion

- Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55-67. Introduce Ridge como estimacion sesgada que estabiliza problemas colineales. Usado en NB04.
- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society, Series B*, 58(1), 267-288. Introduce Lasso, que combina encogimiento y seleccion de variables mediante penalizacion L1. Usado en NB05.
- Zou, H., & Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. *Journal of the Royal Statistical Society, Series B*, 67(2), 301-320. Introduce Elastic Net y su efecto de agrupamiento en variables correlacionadas. Usado en NB06.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization Paths for Generalized Linear Models via Coordinate Descent. *Journal of Statistical Software*, 33(1), 1-22. Presenta rutas de regularizacion eficientes para GLM, base conceptual de glmnet. Usado en NB07.
- Defazio, A., Bach, F., & Lacoste-Julien, S. (2014). SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives. *Advances in Neural Information Processing Systems (NIPS) 27*. Aporta el solver SAGA usado para penaltis L1 y Elastic Net. Usado en NB06.

## Datos desbalanceados y aprendizaje sensible al costo

- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357. Introduce SMOTE, que crea ejemplos sinteticos de la minoria interpolando entre vecinos. Usado en NB08.
- Elkan, C. (2001). The Foundations of Cost-Sensitive Learning. *Proceedings of IJCAI 2001*, 973-978. Explica como costos asimetricos y tasas base cambian la regla optima de decision. Usado en NB08.
- He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284. Revisa problemas de desbalance, muestreo y metricas como ROC y precision-recall. Usado en NB03 y NB08.

## Libros de texto

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Desarrolla regularizacion, sesgo-varianza, validacion cruzada y comparacion L1/L2. Usado en NB04, NB05 y NB07.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Presenta la regresion logistica como clasificador probabilistico discriminativo. Usado en NB01.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Ofrece una perspectiva probabilistica de regularizacion y penaltis combinados. Usado en NB04 y NB06.

## Herramientas

- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830. Describe la biblioteca scikit-learn usada para modelos, pipelines, metricas y validacion cruzada. Usado en NB07.
