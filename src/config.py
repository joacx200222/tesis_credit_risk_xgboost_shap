from pathlib import Path

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

RAW_DTA_PATH = BASE_DIR / "BD_Encuesta_Capacidades_Financieras_2022.dta"
RAW_DATASET_CSV = RAW_DATA_DIR / "dataset_sbs_caf_2022.csv"
PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_VAL_PATH = PROCESSED_DATA_DIR / "val.csv"
PROCESSED_TEST_PATH = PROCESSED_DATA_DIR / "test.csv"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.json"
XGBOOST_MODEL_JSON = MODELS_DIR / "xgboost_credit_model.json"
BENCHMARK_RESULTS_PATH = MODELS_DIR / "benchmark_results.json"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"
SHAP_SUMMARY_PATH = MODELS_DIR / "shap_summary_stats.json"

# Partición y reproducibilidad
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15  # test = el 15% restante

# Variable objetivo: pagó tarde o falló en sus pagos (pregunta F12_16)
TARGET_COLUMN = "default"  # 1 = moroso, 0 = cumplidor
SCALE_POS_WEIGHT = 2.99    # 900 / 301 en la base completa; al entrenar se recalcula con train (630 / 210)

# Variables por eje

# Eje 1: sociodemográfico y ocupación
NUMERICAL_SOCIODEMO_FEATURES = [
    "edad",
    "carga_familiar_adultos",
    "carga_familiar_ninos",
]

CATEGORICAL_SOCIODEMO_FEATURES = [
    "genero",               # "Hombre", "Mujer"
    "nivel_educacion",      # Primaria, Secundaria, Técnico, Universitario
    "departamento",         # Departamento de residencia (Perú)
    "ambito",               # "Urbano", "Rural"
    "condicion_laboral",    # Independiente / Cuenta propia, Empleado, Empleador, etc.
    "es_informal",          # "Informal / Sin RUC", "Formalizado / Con RUC o Beneficios"
]

# Eje 2: inclusión digital y datos alternativos
NUMERICAL_DIGITAL_FEATURES = [
    "frecuencia_yape_plin",             # 1: Nunca, 2: Poco, 3: Casi siempre, 4: Siempre
    "pago_celular_frecuencia",          # 1 a 5: Uso celular para cobros y pagos
    "uso_whatsapp",                     # 1 a 4: Frecuencia de mensajería instantánea
    "uso_internet",                     # 1 a 4: Frecuencia de navegación / internet
    "preferencia_efectivo",             # 1 a 5: Grado de acuerdo en que efectivo es más seguro que medios digitales
    "confianza_herramientas_digitales", # 1 a 5: Facilidad de finanzas digitales
    "puntualidad_pago_deudas",          # 1 a 5: Pago puntual de obligaciones financieras
]

CATEGORICAL_DIGITAL_FEATURES = [
    "adopcion_yape_plin",         # "Sí", "No" (Conoce o usa Yape/Plin/BIM)
    "tiene_billetera_activa",     # "Sí", "No" (Tenencia activa)
    "guarda_dinero_recibos",      # "Sí", "No" (Reserva fondos para facturas y servicios)
    "anota_recibos_vencer",       # "Sí", "No" (Registra vencimientos de luz/agua/teléfono)
]

# Eje 3: capacidad y comportamiento financiero
NUMERICAL_CAPACITY_FEATURES = [
    "dificultad_emergencia_1000",   # 1 a 4: Dificultad para reunir S/ 1,000 ante emergencia
    "meses_resiliencia_desempleo",  # 1 (<1 sem) a 5 (>6 meses): Resiliencia ante desempleo
    "sobra_dinero_fin_mes",         # 1 a 5: Dinero sobrante al final del mes
    "ingreso_apenas_alcanza",       # 1 a 5: Vulnerabilidad de liquidez cotidiana
    "percepcion_sobreendeudado",    # 1 a 5: Sensación de tener demasiadas deudas
]

CATEGORICAL_CAPACITY_FEATURES = [
    "rango_ingreso",                # Tramo de ingresos mensuales en Soles
    "nivel_socioeconomico",         # NSE A, B, C, D, E
    "ingresos_estables",            # "Sí", "No" (Regularidad de ingresos)
    "planifica_presupuesto",        # "Sí", "No" (Plan de ingresos y gastos)
    "anota_gastos",                 # "Sí", "No" (Registro de egresos)
    "capacidad_afrontar_imprevisto",# "Sí", "No" (Afronta imprevisto sin endeudarse)
    # crédito formal e informal
    "tiene_cuenta_ahorro",          # "Sí", "No"
    "tiene_tarjeta_credito",        # "Sí", "No"
    "tiene_prestamo_personal",      # "Sí", "No"
    "ha_usado_prestamista",         # "Sí", "No" (Crédito informal / gota a gota)
    "ha_usado_pandero_junta",       # "Sí", "No" (Crédito informal colaborativo)
    "credito_formal_negado",        # "Sí", "No" (Rechazo previo en banca/cajas)
]

# Índices calculados en feature_engineering.py (0 a 1)
NUMERICAL_DERIVED_FEATURES = [
    "indice_inclusion_digital",
    "indice_resiliencia_financiera",
    "indice_vulnerabilidad_liquidez",
    "indice_cultura_pago",
]

ALL_NUMERICAL_FEATURES = (
    NUMERICAL_SOCIODEMO_FEATURES +
    NUMERICAL_DIGITAL_FEATURES +
    NUMERICAL_CAPACITY_FEATURES +
    NUMERICAL_DERIVED_FEATURES
)

ALL_CATEGORICAL_FEATURES = (
    CATEGORICAL_SOCIODEMO_FEATURES +
    CATEGORICAL_DIGITAL_FEATURES +
    CATEGORICAL_CAPACITY_FEATURES
)


# Score y política de crédito
SCORE_MIN = 0
SCORE_MAX = 1000

DEFAULT_LGD = 0.50  # para la pérdida esperada EL = PD * EAD * LGD

# Score = 1000 * (1 - PD). Aprueba si PD <= 0.30, revisión manual hasta 0.45, rechaza por encima.
THRESHOLD_LOW_RISK = 0.30
THRESHOLD_MEDIUM_RISK = 0.45

# Opciones de los selectores de la app
OPCIONES_DEPARTAMENTO = [
    "Lima / Callao", "Arequipa", "Cusco", "Piura", "La Libertad",
    "Junín", "Puno", "Lambayeque", "San Martín", "Áncash", "Ica",
    "Ayacucho", "Cajamarca", "Loreto", "Huánuco", "Tacna", "Otros Departamentos"
]

OPCIONES_EDUCACION = [
    "Primaria / Sin Instrucción",
    "Secundaria Completa",
    "Superior Técnico Completo",
    "Superior Universitario Completo"
]

OPCIONES_CONDICION_LABORAL = [
    "Trabajador Independiente / Cuenta propia",
    "Empleado Técnico o Profesional",
    "Obrero / Empleado Operativo",
    "Empleador o Patrono (con personal)",
    "Campesino / Agricultor / Ganadero",
    "Comerciante Independiente"
]

OPCIONES_RANGO_INGRESO = [
    "Menor a S/ 600",
    "De S/ 601 a S/ 1,200 (Ingreso Mínimo)",
    "De S/ 1,201 a S/ 2,400",
    "De S/ 2,401 a S/ 4,000",
    "Mayor a S/ 4,000"
]

OPCIONES_NSE = ["A/B", "C", "D", "E"]

# Hiperparámetros base. Los optimizados salen de hyperparameter_tuning.py (models/best_params.json).
XGBOOST_DEFAULT_PARAMS = {
    "n_estimators": 180,
    "max_depth": 4,
    "learning_rate": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 2,
    "gamma": 0.15,
    "reg_alpha": 0.3,   # L1
    "reg_lambda": 1.5,  # L2
    "scale_pos_weight": SCALE_POS_WEIGHT,
    "eval_metric": "auc",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RF_DEFAULT_PARAMS = {
    "n_estimators": 150,
    "max_depth": 5,
    "min_samples_split": 5,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

LOGREG_DEFAULT_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

# Búsqueda con Optuna
CV_FOLDS = 5
OPTUNA_TRIALS = {"XGBoost": 100, "Random_Forest": 60, "Regresion_Logistica": 30}
