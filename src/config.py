"""
==============================================================================
PROYECTO DE TESIS - UNIVERSIDAD DE LIMA
Carrera: Ingeniería de Sistemas
Título: Modelo de evaluación de riesgo crediticio para microfinanzas e
        inclusión financiera mediante XGBoost e Inteligencia Artificial
        Explicable (SHAP)
Archivo: src/config.py
Descripción: Configuración centralizada, constantes de negocio y rutas del sistema.
==============================================================================
"""

from pathlib import Path

# ------------------------------------------------------------------------------
# 1. RUTAS DEL PROYECTO (CRISP-DM Structure)
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

# Archivos específicos
RAW_DATASET_PATH = RAW_DATA_DIR / "dataset_microfinanzas_peru.csv"
PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_VAL_PATH = PROCESSED_DATA_DIR / "val.csv"
PROCESSED_TEST_PATH = PROCESSED_DATA_DIR / "test.csv"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_credit_model.joblib"
XGBOOST_MODEL_JSON = MODELS_DIR / "xgboost_credit_model.json"
BENCHMARK_RESULTS_PATH = MODELS_DIR / "benchmark_results.json"
SHAP_SUMMARY_PATH = MODELS_DIR / "shap_summary_stats.joblib"

# ------------------------------------------------------------------------------
# 2. PARÁMETROS GENERALES Y REPRODUCIBILIDAD
# ------------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
TRAIN_SIZE = 0.70

# Variable Objetivo (Target)
TARGET_COLUMN = "default_90d"  # 1: Incumplimiento >= 90 días, 0: Buen pagador

# ------------------------------------------------------------------------------
# 3. DEFINICIÓN DE CARACTERÍSTICAS (FEATURE DEFINITIONS)
# ------------------------------------------------------------------------------

# Variables Numéricas Tradicionales
NUMERICAL_TRADITIONAL_FEATURES = [
    "edad",
    "antiguedad_negocio_meses",
    "ingreso_mensual_neto",
    "gasto_mensual_operativo",
    "monto_solicitado",
    "plazo_meses",
    "tasa_interes_anual",
    "num_consultas_centrales_ult_6m",
]

# Variables Numéricas Alternativas / Thin-File (Inclusión Financiera ODS 10)
NUMERICAL_ALTERNATIVE_FEATURES = [
    "puntualidad_servicios_basicos",      # Ratio 0.0 a 1.0 de puntualidad en luz/agua
    "volumen_transacciones_digitales",    # Monto mensual en billeteras digitales (Yape/Plin) en S/
    "frecuencia_transacciones_yape_mes",  # Número de transacciones mensuales entrantes
    "antiguedad_celular_meses",           # Meses con la misma línea telefónica móvil
    "variabilidad_ingresos_cv",           # Coeficiente de variación de ventas (estacionalidad)
]

# Ratios Financieros Derivados (Feature Engineering)
NUMERICAL_DERIVED_FEATURES = [
    "cuota_mensual",
    "flujo_caja_libre_mensual",
    "ratio_cuota_ingreso",                # Debt-to-Income (DTI)
    "ratio_cobertura_deuda",              # Debt Service Coverage Ratio (DSCR)
    "ratio_monto_ingreso",                # Monto / Ingreso Neto
    "indice_estabilidad_negocio",         # Combinación antigüedad y baja variabilidad
    "score_comportamiento_alternativo",   # Score compuesto de datos alternativos
]

ALL_NUMERICAL_FEATURES = (
    NUMERICAL_TRADITIONAL_FEATURES +
    NUMERICAL_ALTERNATIVE_FEATURES +
    NUMERICAL_DERIVED_FEATURES
)

# Variables Categóricas
CATEGORICAL_FEATURES = [
    "genero",
    "estado_civil",
    "nivel_educacion",
    "tipo_vivienda",
    "region_peru",
    "rubro_negocio",
    "es_informal",                        # 1: Negocio no bancarizado/RUC 10, 0: Formal
    "posee_garantia_o_aval",              # 1: Sí, 0: No
]

ALL_MODEL_FEATURES = ALL_NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ------------------------------------------------------------------------------
# 4. PARÁMETROS FINANCIEROS Y DE NEGOCIO (Contexto Microfinanciero Peruano / SBS)
# ------------------------------------------------------------------------------
# Parámetros de Pérdida Esperada (Expected Loss: EL = PD * EAD * LGD)
DEFAULT_LGD = 0.50  # Loss Given Default promedio en microfinanzas (50%)

# Umbrales de Decisión Crediticia (Scorecard Cut-offs)
THRESHOLD_LOW_RISK = 0.20       # PD < 0.20: Aprobado automático (Bajo Riesgo)
THRESHOLD_MEDIUM_RISK = 0.45    # 0.20 <= PD <= 0.45: Revisión manual (Riesgo Moderado)
# PD > 0.45: Rechazado automático (Alto Riesgo)

# Escala de Score Crediticio (0 a 1000 puntos)
SCORE_MIN = 0
SCORE_MAX = 1000

# Opciones válidas para interfaz y generación de datos
REGIONES_PERU = [
    "Lima/Callao", "Arequipa", "Cusco", "Piura", "La Libertad",
    "Junín", "Puno", "Lambayeque", "San Martín", "Áncash", "Ica"
]

RUBROS_NEGOCIO = [
    "Comercio Minorista / Bodega",
    "Gastronomía / Restaurante",
    "Servicios Técnicos / Profesionales",
    "Transporte / Mototaxi / Taxi",
    "Manufactura / Confección Textil",
    "Agropecuario / Comercialización Rural",
    "Ferretería / Materiales de Construcción"
]

NIVELES_EDUCACION = [
    "Primaria Completa",
    "Secundaria Completa",
    "Técnico Superior",
    "Universitario Incompleto",
    "Universitario Completo",
    "Postgrado"
]

TIPOS_VIVIENDA = [
    "Propia",
    "Familiar",
    "Alquilada",
    "Hipotecada"
]

ESTADOS_CIVILES = [
    "Soltero(a)",
    "Casado(a)",
    "Conviviente",
    "Divorciado(a)",
    "Viudo(a)"
]

# ------------------------------------------------------------------------------
# 5. HIPERPARÁMETROS DEL MODELO XGBOOST
# ------------------------------------------------------------------------------
XGBOOST_DEFAULT_PARAMS = {
    "n_estimators": 250,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "auc",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
