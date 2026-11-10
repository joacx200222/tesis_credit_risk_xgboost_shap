# Modelo de Evaluación de Riesgo Crediticio para Microfinanzas mediante XGBoost e IA Explicable (TreeSHAP)

[![tests](https://github.com/joacx200222/tesis_credit_risk_xgboost_shap/actions/workflows/tests.yml/badge.svg)](https://github.com/joacx200222/tesis_credit_risk_xgboost_shap/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-cost--sensitive-189FDD)
![SHAP](https://img.shields.io/badge/SHAP-TreeSHAP-FF0D57)
![Optuna](https://img.shields.io/badge/Optuna-TPE-4A6EE0)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **In short (EN):** A credit-scoring pipeline for Peruvian microfinance that estimates each applicant's probability of default, including *thin-file* borrowers with no credit-bureau history. It uses alternative data from the SBS/CAF 2022 National Financial Capability Survey (mobile-wallet use, bill-payment habits, financial resilience), tunes and compares Logistic Regression, Random Forest and a cost-sensitive XGBoost with Optuna, and explains every decision with TreeSHAP, both portfolio-wide and per applicant, turning the main risk factors into plain-language denial reasons. XGBoost reaches **AUC 0.756** and **KS 44.1%** on a held-out test set. It ships with a Streamlit app for scoring individual applications.

**Tesista:** Joaquín Ignacio Bernales Demichelli — Ingeniería de Sistemas, Universidad de Lima

## Arquitectura

```mermaid
flowchart LR
    DTA[("Encuesta SBS/CAF 2022<br/>.dta")] --> ING["ingenieria_variables<br/>variable objetivo + 4 índices"]
    ING --> PRE["preprocesamiento<br/>imputación, escalado, One-Hot<br/>partición 70/15/15"]
    PRE --> CSV[("data/processed<br/>train / val / test")]
    PRE --> PJ[("preprocesador.json")]
    CSV --> BUS["busqueda_hiperparametros<br/>Optuna TPE + CV 5 folds"]
    BUS --> MP[("mejores_params.json")]
    CSV --> ENT["entrenamiento<br/>LR vs RF vs XGBoost"]
    MP --> ENT
    ENT --> EVA["evaluacion<br/>AUC, KS, PR-AUC, F1"]
    ENT --> MOD[("modelo_xgboost.json")]
    EVA --> BEN[("benchmark.json")]
    MOD --> EXP["explicabilidad<br/>TreeSHAP local y global"]
    PJ --> EXP
    EXP --> APP["app (Streamlit)<br/>score, PD, decisión y motivos"]
    BEN --> APP
```

## Descripción

Pipeline de aprendizaje supervisado que estima la probabilidad de mora (PD) de solicitantes de microcrédito, incluidos prestatarios *thin-file* (sin historial en centrales de riesgo), y explica cada decisión. Resuelve dos limitaciones del scoring tradicional:

1. **Exclusión del thin-file:** usa variables alternativas de la **Encuesta Nacional de Capacidades Financieras SBS/CAF 2022** (1,201 encuestados, 24 departamentos): adopción de billeteras móviles (Yape/Plin/BIM), hábitos de pago de recibos, resiliencia ante imprevistos y vulnerabilidad de liquidez.
2. **Caja negra:** integra **TreeSHAP** para explicar la decisión a nivel global (portafolio) y local (solicitud individual), y traducir los factores de riesgo en razones de acción adversa que permiten informar los motivos de denegación que exige la Ley N.° 32294 (2025).

## Resultados (conjunto de prueba independiente, 181 registros)

| Modelo | AUC-ROC | KS por deciles (%) | ¿KS ≥ 30%? |
| :--- | :---: | :---: | :---: |
| Regresión Logística (scorecard tradicional) | 0.6670 | 28.28 | No |
| Random Forest | 0.7064 | 32.43 | Sí |
| **XGBoost sensible al costo (propuesto)** | **0.7557** | **44.09** | **Sí** |

Los tres modelos usan hiperparámetros optimizados con Optuna (validación cruzada de 5 folds sobre train). Con los hiperparámetros base fijos, los resultados en test eran: Regresión Logística 0.6799 / 29.95%, Random Forest 0.7240 / 35.78% y XGBoost 0.7134 / 33.67%. La optimización mejoró a XGBoost y bajó levemente a los otros dos en test, aunque los tres mejoraron en validación cruzada.

Detalle completo (PR-AUC, F1, recall, matriz de confusión) en `models/benchmark.json`. Inferencia + explicación TreeSHAP: ~30–50 ms por solicitud.

## Estructura del repositorio (CRISP-DM)

```text
tesis_credit_risk_xgboost_shap/
├── BD_Encuesta_Capacidades_Financieras_2022.dta   # Base oficial SBS/CAF 2022 (no se sube, ver NOTICE.md)
├── data/
│   ├── raw/dataset_sbs_caf_2022.csv               # Base extraída y enriquecida (se genera, no se sube)
│   └── processed/{train,val,test}.csv             # Partición estratificada 70/15/15 (840/180/181)
├── models/
│   ├── preprocesador.json                         # Preprocesador ajustado (94 variables de salida)
│   ├── modelo_xgboost{,_nativo}.json              # Metadatos + modelo nativo de XGBoost
│   ├── mejores_params.json                        # Mejores hiperparámetros encontrados por Optuna
│   ├── benchmark.json                             # Comparativa de los 3 modelos (base vs optimizado)
│   ├── resumen_shap.json                          # Valor base e importancia global SHAP
│   └── shap_beeswarm.png                          # Explicabilidad global
├── src/
│   ├── config.py                 # Rutas, variables, hiperparámetros y política de crédito
│   ├── ingenieria_variables.py   # Ingesta del .dta, variable objetivo y 4 índices sintéticos
│   ├── preprocesamiento.py       # Imputación, escalado robusto, One-Hot y partición estratificada
│   ├── busqueda_hiperparametros.py # Optuna (TPE) + validación cruzada estratificada de 5 folds
│   ├── entrenamiento.py          # Regresión Logística vs Random Forest vs XGBoost
│   ├── evaluacion.py             # AUC-ROC, PR-AUC, KS, F1, pérdida esperada, score y decisión
│   ├── explicabilidad.py         # TreeSHAP, razones de acción adversa y dictamen
│   └── app.py                    # Prototipo web en Streamlit
├── tests/
│   ├── test_pipeline.py              # Pruebas del pipeline completo y de TreeSHAP
│   └── test_busqueda_hiperparametros.py # Validación cruzada y búsqueda Optuna
├── .github/workflows/tests.yml   # CI: corre las pruebas en cada push a main
├── Dockerfile
├── LICENSE                       # MIT
├── NOTICE.md                     # Datos y librerías de terceros
└── requirements.txt
```

## Metodología por fase CRISP-DM

| Fase | Módulo | Qué hace |
| :--- | :--- | :--- |
| 1. Comprensión del negocio | `config.py` | Piso KS ≥ 30%, LGD = 50%, política: PD ≤ 0.30 aprueba, ≤ 0.45 revisión manual, > 0.45 rechaza. |
| 2. Comprensión de los datos | `ingenieria_variables.py` | Lee el `.dta` oficial y mapea las preguntas del cuestionario a 37 variables en 3 ejes. |
| 3. Preparación de los datos | `ingenieria_variables.py`, `preprocesamiento.py` | Variable objetivo, 4 índices, imputación por mediana/moda, escalado mediana/IQR, One-Hot (94 variables) y partición estratificada 70/15/15. |
| 4. Modelado | `busqueda_hiperparametros.py`, `entrenamiento.py` | Optimización de hiperparámetros de los 3 modelos con Optuna (búsqueda bayesiana TPE; XGBoost 100 intentos, Random Forest 60, Regresión Logística 30), maximizando el AUC promedio de una validación cruzada estratificada de 5 folds solo sobre train. Luego se entrena XGBoost con `scale_pos_weight` = negativos/positivos de train (≈ 3.0) frente a Regresión Logística y Random Forest balanceados. |
| 5. Evaluación | `evaluacion.py`, `explicabilidad.py` | Métricas en el set de prueba y descomposición TreeSHAP. |
| 6. Despliegue | `app.py` | Originación individual, explicabilidad global y benchmark. |

### Variable objetivo (mora)

`default = 1` si el encuestado respondió que, ante la falta de ingresos, **pagó sus cuentas tarde o falló en sus pagos** (pregunta `F12_16`). Quienes nunca tuvieron déficit (`F11`) no responden esa pregunta y se asignan a `default = 0`. Resultado: 301 morosos (25.06%) y 900 cumplidores.

### Índices sintéticos (0 a 1)

Sumas ponderadas de respuestas normalizadas; los pesos se definieron por criterio de dominio.

| Índice | Componentes (peso) |
| :--- | :--- |
| Inclusión Digital | usa billetera (0.30), frecuencia de uso (0.30), pagos por celular (0.20), WhatsApp (0.10), internet (0.10) |
| Resiliencia Financiera | afronta imprevisto sin endeudarse (0.35), meses ante desempleo (0.25), dinero sobrante (0.20), facilidad de reunir S/ 1,000 (0.20) |
| Vulnerabilidad de Liquidez | ingreso apenas alcanza (0.45), sobreendeudamiento (0.40), crédito negado (0.15) |
| Cultura de Pago | puntualidad (0.50), guarda dinero para recibos (0.25), anota vencimientos (0.25) |

### Fórmulas

- **Pérdida esperada:** $EL = PD \times EAD \times LGD$, con $EAD$ = monto solicitado y $LGD = 50\%$.
- **Score:** $Score = 1000 \times (1 - PD)$.
- **Kolmogorov-Smirnov:** $KS = \max |F_{\text{malos}}(s) - F_{\text{buenos}}(s)|$, calculado sobre 10 deciles de riesgo.
- **Aditividad TreeSHAP:** $f(x) = E[f(x)] + \sum_i \phi_i(x)$ en log-odds (verificada en las pruebas con error < $10^{-4}$).

## Instalación y ejecución

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt

# Opcional: regenerar todo desde la encuesta. Requiere el .dta en la raíz (ver NOTICE.md);
# sin él, el repo ya trae los CSV procesados y los modelos entrenados.
python -m src.preprocesamiento         # Ingesta del .dta, índices, partición y preprocesador (~4 s)
python -m src.busqueda_hiperparametros # Búsqueda Optuna + CV 5-fold (~10 min; genera models/mejores_params.json)
python -m src.entrenamiento            # Entrena y compara los 3 modelos, base vs optimizado (~10 s)
python -m src.explicabilidad           # Genera el beeswarm global y las estadísticas SHAP

python -m unittest discover tests -v   # funciona con o sin el .dta
streamlit run src/app.py               # http://localhost:8501
```

### Docker

```bash
docker build -t credit-risk-xgboost-shap .
docker run -p 8501:8501 credit-risk-xgboost-shap
```

## Licencia

El código está bajo licencia [MIT](LICENSE). Los datos de la encuesta SBS/CAF y las librerías usadas tienen sus propias condiciones; el detalle está en [NOTICE.md](NOTICE.md).

## Referencias principales

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794.
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD*, 2623–2631. https://doi.org/10.1145/3292500.3330701
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67.
- Bussmann, N., Giudici, P., Marinelli, D., & Papenbrock, J. (2021). Explainable machine learning in credit risk management. *Computational Economics*, 57(1), 203–216.
- Elkan, C. (2001). The foundations of cost-sensitive learning. *IJCAI*, 973–978.
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284.
- Shapley, L. S. (1953). A value for n-person games. *Contributions to the Theory of Games*, 2, 307–317.
- Superintendencia de Banca, Seguros y AFP. (2011). *Resolución SBS N.° 3780-2011: Reglamento de la Gestión de Riesgo de Crédito* (y modificatorias).
- Congreso de la República del Perú. (2025). *Ley N.° 32294*, que modifica la Ley N.° 28587 para obligar a informar los motivos de denegatoria de créditos.
- Superintendencia de Banca, Seguros y AFP. (2025). *Resolución SBS N.° 01689-2025*.
- SBS & CAF. (2022). *Encuesta Nacional de Capacidades Financieras del Perú 2022*.
