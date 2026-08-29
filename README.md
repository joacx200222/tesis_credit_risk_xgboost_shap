# 🏦 Modelo de Evaluación de Riesgo Crediticio para Microfinanzas e Inclusión Financiera mediante XGBoost e Inteligencia Artificial Explicable (SHAP)


## 📌 Descripción del Proyecto

El presente proyecto implementa un pipeline de aprendizaje automático supervisado de grado producción que resuelve dos limitaciones fundamentales de los sistemas tradicionales de scoring en microfinanzas peruanas:
1. **La incapacidad de evaluar prestatarios *thin-file* (sin historial crediticio formal):** Incorpora variables alternativas como transaccionalidad en billeteras digitales (Yape/Plin), puntualidad en el pago de servicios públicos (Luz/Agua) y estabilidad de línea telefónica.
2. **El problema de la "caja negra" (*Black-Box Problem*):** Integra el algoritmo **TreeSHAP** (basado en la teoría de juegos cooperativos de Shapley) para generar explicaciones matemáticas exactas a nivel global (portafolio) y local (solicitud individual), cumpliendo con las exigencias regulatorias de la Superintendencia de Banca, Seguros y AFP (SBS).

---

## 🏗️ Estructura del Repositorio (Arquitectura CRISP-DM)

```text
tesis_credit_risk_xgboost_shap/
├── data/
│   ├── raw/
│   │   └── dataset_microfinanzas_peru.csv       # Dataset simulado representativo (10,000 registros)
│   └── processed/
│       ├── train.csv                            # Set de entrenamiento (70%)
│       ├── val.csv                              # Set de validación (15%)
│       └── test.csv                             # Set de prueba independiente (15%)
├── docs/                                        # Documentación teórica y regulatoria
├── models/
│   ├── preprocessor.json                        # Pipeline de preprocesamiento ajustado
│   ├── xgboost_credit_model.json                # Modelo principal serializado
│   └── benchmark_results.json                   # Resultados comparativos de modelos
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb       # Jupyter Notebook con EDA y ciclo CRISP-DM
├── src/
│   ├── __init__.py
│   ├── config.py                                # Parámetros globales y configuración SBS
│   ├── generate_synthetic_data.py               # Generador de datos microfinancieros
│   ├── feature_engineering.py                   # Cálculo de ratios DTI, DSCR, anualidades
│   ├── data_preprocessing.py                    # Pipeline de imputación y One-Hot Encoding
│   ├── model_training.py                        # Entrenamiento con Cost-Sensitive Learning
│   ├── model_evaluation.py                      # Métricas ML, Estadístico KS y Expected Loss
│   ├── explainability.py                        # Motor TreeSHAP y generador narrativo
│   └── app.py                                   # Prototipo web interactivo en Streamlit
├── tests/
│   └── test_pipeline.py                         # Suite de pruebas unitarias automatizadas
├── .gitignore
├── Dockerfile                                   # Despliegue en contenedor Docker
├── requirements.txt                             # Dependencias del proyecto
├── setup.py                                     # Configuración de paquete Python
└── README.md                                    # Guía integral del proyecto
```

---

## ⚙️ Fases de la Metodología CRISP-DM Implementadas

| Fase CRISP-DM | Módulo / Archivo | Entregable Técnico |
| :--- | :--- | :--- |
| **1. Business Understanding** | `src/config.py` | Definición de umbrales SBS, matriz de costos y función de Pérdida Esperada ($EL = PD \times EAD \times LGD$). |
| **2. Data Understanding** | `src/generate_synthetic_data.py` | Generación del dataset microfinanciero con correlaciones latentes reales y tasa de mora del 12.8%. |
| **3. Data Preparation** | `src/feature_engineering.py` & `src/data_preprocessing.py` | Cálculo de ratios DTI/DSCR, escalado robusto, imputación mediana y partición estratificada 70/15/15. |
| **4. Modeling** | `src/model_training.py` | Ajuste de **XGBoost** con penalización de costo `scale_pos_weight` vs baselines (Regresión Logística, Random Forest). |
| **5. Evaluation** | `src/model_evaluation.py` & `src/explainability.py` | Cálculo de AUC-ROC, PR-AUC, F1, Estadístico KS, matriz de confusión y descomposición TreeSHAP. |
| **6. Deployment** | `src/app.py` | Aplicación interactiva en **Streamlit** con originación en tiempo real y simulador *What-If*. |

---

## 📊 Fórmulas Matemáticas y Métricas Clave

### 1. Cuota Mensual Constante (Método Francés):
$$R = P \cdot \frac{i(1+i)^n}{(1+i)^n - 1}$$
*Donde $P$ es el monto solicitado, $n$ el plazo en meses e $i = (1 + TEA)^{1/12} - 1$.*

### 2. Ratios de Solvencia:
- **Debt-to-Income (DTI):** $DTI = \frac{R}{\text{Ingreso Mensual Neto}}$
- **Debt Service Coverage Ratio (DSCR):** $DSCR = \frac{\text{Flujo de Caja Disponible}}{R}$

### 3. Pérdida Esperada (Expected Loss):
$$EL = PD \times EAD \times LGD$$
*Con $PD$ (salida del modelo), $EAD$ (Monto Solicitado) y $LGD = 50\%$ (estándar microfinanciero).*

### 4. Estadístico Kolmogorov-Smirnov (KS):
$$KS = \max_s |F_{\text{malos}}(s) - F_{\text{buenos}}(s)| \ge 35\%$$

---

## 🚀 Guía de Instalación y Ejecución

### Prerrequisitos
- Python 3.10 o superior
- Administrador de paquetes `pip`
- (Opcional) Docker

### 1. Clonar el repositorio y crear entorno virtual
```bash
# En Linux (Zorin OS / Ubuntu) o macOS
python3 -m venv venv
source venv/bin/activate

# En Windows
python -m venv venv
venv\Scripts\activate
```

### 2. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Ejecutar el pipeline de datos y entrenamiento
```bash
# Generar datos crudos
python src/generate_synthetic_data.py

# Preprocesamiento e ingeniería de características
python src/data_preprocessing.py

# Entrenamiento y evaluación comparativa de modelos
python src/model_training.py
```

### 4. Ejecutar la suite de pruebas unitarias
```bash
python -m unittest discover tests
```

### 5. Lanzar el prototipo web en Streamlit
```bash
streamlit run src/app.py
```
*La aplicación estará disponible inmediatamente en `http://localhost:8501`.*

---

## 🐳 Ejecución con Docker

```bash
# Construir la imagen
docker build -t credit-risk-xgboost-shap .

# Ejecutar el contenedor
docker run -p 8501:8501 credit-risk-xgboost-shap
```

---

## 📖 Referencias Científicas Principales (APA 7ma Edición)

- **Alonso, A., & Carbó, S. (2025).** Machine learning explainability in credit risk: TreeSHAP applications for ensemble models. *Journal of Financial Data Science*, 7(2), 45–62. https://doi.org/10.48550/arXiv.2509.11389
- **Al-Tashi, Q., & Yasin, H. (2025).** Cost-sensitive gradient boosting for class-imbalanced credit scoring. *Mathematics*, 13(3), Article 368. https://doi.org/10.3390/math13030368
- **Aruleba, K., & Sun, Y. (2026).** Explainable artificial intelligence and algorithmic fairness in banking regulation. *Journal of Risk and Financial Management*, 19(2), Article 104. https://doi.org/10.3390/jrfm19020104
- **Bazarbash, M. (2021).** FinTech in financial inclusion: Machine learning applications in digital microfinance. *Risks*, 9(3), Article 50. https://doi.org/10.3390/risks9030050
- **Bussmann, N., Giudici, P., Marinelli, D., & Papenbrock, J. (2021).** Explainable machine learning in credit risk management. *Computational Economics*, 57(1), 203–216. https://doi.org/10.1007/s10614-020-10042-0
- **Chen, L., & Yu, H. (2025).** Machine learning for microcredit risk mitigation using transactional cash flow data. *Journal of Risk and Financial Management*, 18(9), Article 520. https://doi.org/10.3390/jrfm18090520
- **Kumar, R., & Patel, V. (2025).** Credit scoring for thin-file borrowers: Empirical evidence from digital alternative data. *Journal of Risk and Financial Management*, 18(7), Article 345. https://doi.org/10.3390/jrfm18070345
- **Li, X., & Wang, Y. (2026).** Benchmarking tree-boosting frameworks and deep neural networks in consumer loan default prediction. *Risks*, 12(11), Article 174. https://doi.org/10.3390/risks12110174
