from pathlib import Path

# Rutas
DIR_BASE = Path(__file__).resolve().parent.parent
DIR_DATOS = DIR_BASE / "data"
DIR_CRUDOS = DIR_DATOS / "raw"
DIR_PROCESADOS = DIR_DATOS / "processed"
DIR_MODELOS = DIR_BASE / "models"

RUTA_DTA = DIR_BASE / "BD_Encuesta_Capacidades_Financieras_2022.dta"
RUTA_CSV_COMPLETO = DIR_CRUDOS / "dataset_sbs_caf_2022.csv"
RUTA_TRAIN = DIR_PROCESADOS / "train.csv"
RUTA_VAL = DIR_PROCESADOS / "val.csv"
RUTA_TEST = DIR_PROCESADOS / "test.csv"

RUTA_PREPROCESADOR = DIR_MODELOS / "preprocesador.json"
RUTA_MODELO = DIR_MODELOS / "modelo_xgboost.json"
RUTA_BENCHMARK = DIR_MODELOS / "benchmark.json"
RUTA_MEJORES_PARAMS = DIR_MODELOS / "mejores_params.json"
RUTA_RESUMEN_SHAP = DIR_MODELOS / "resumen_shap.json"

# Partición y reproducibilidad
SEMILLA = 42
PROP_TRAIN = 0.70
PROP_VAL = 0.15  # test = el 15% restante

# Variable objetivo: pagó tarde o falló en sus pagos (pregunta F12_16)
COLUMNA_OBJETIVO = "default"  # 1 = moroso, 0 = cumplidor
SCALE_POS_WEIGHT = 2.99    # 900 / 301 en la base completa; al entrenar se recalcula con train (630 / 210)

# Variables por eje

# Eje 1: sociodemográfico y ocupación
NUMERICAS_SOCIODEMO = [
    "edad",
    "carga_familiar_adultos",
    "carga_familiar_ninos",
]

CATEGORICAS_SOCIODEMO = [
    "genero",               # "Hombre", "Mujer"
    "nivel_educacion",      # Primaria, Secundaria, Técnico, Universitario
    "departamento",         # Departamento de residencia (Perú)
    "ambito",               # "Urbano", "Rural"
    "condicion_laboral",    # Independiente / Cuenta propia, Empleado, Empleador, etc.
    "es_informal",          # "Informal / Sin RUC", "Formalizado / Con RUC o Beneficios"
]

# Eje 2: inclusión digital y datos alternativos
NUMERICAS_DIGITAL = [
    "frecuencia_yape_plin",             # 1: Nunca, 2: Poco, 3: Casi siempre, 4: Siempre
    "pago_celular_frecuencia",          # 1 a 5: Uso celular para cobros y pagos
    "uso_whatsapp",                     # 1 a 4: Frecuencia de mensajería instantánea
    "uso_internet",                     # 1 a 4: Frecuencia de navegación / internet
    "preferencia_efectivo",             # 1 a 5: Grado de acuerdo en que efectivo es más seguro que medios digitales
    "confianza_herramientas_digitales", # 1 a 5: Facilidad de finanzas digitales
    "puntualidad_pago_deudas",          # 1 a 5: Pago puntual de obligaciones financieras
]

CATEGORICAS_DIGITAL = [
    "adopcion_yape_plin",         # "Sí", "No" (Conoce o usa Yape/Plin/BIM)
    "tiene_billetera_activa",     # "Sí", "No" (Tenencia activa)
    "guarda_dinero_recibos",      # "Sí", "No" (Reserva fondos para facturas y servicios)
    "anota_recibos_vencer",       # "Sí", "No" (Registra vencimientos de luz/agua/teléfono)
]

# Eje 3: capacidad y comportamiento financiero
NUMERICAS_CAPACIDAD = [
    "dificultad_emergencia_1000",   # 1 a 4: Dificultad para reunir S/ 1,000 ante emergencia
    "meses_resiliencia_desempleo",  # 1 (<1 sem) a 5 (>6 meses): Resiliencia ante desempleo
    "sobra_dinero_fin_mes",         # 1 a 5: Dinero sobrante al final del mes
    "ingreso_apenas_alcanza",       # 1 a 5: Vulnerabilidad de liquidez cotidiana
    "percepcion_sobreendeudado",    # 1 a 5: Sensación de tener demasiadas deudas
]

CATEGORICAS_CAPACIDAD = [
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

# Índices calculados en ingenieria_variables.py (0 a 1)
INDICES = [
    "indice_inclusion_digital",
    "indice_resiliencia_financiera",
    "indice_vulnerabilidad_liquidez",
    "indice_cultura_pago",
]

VARIABLES_NUMERICAS = (
    NUMERICAS_SOCIODEMO +
    NUMERICAS_DIGITAL +
    NUMERICAS_CAPACIDAD +
    INDICES
)

VARIABLES_CATEGORICAS = (
    CATEGORICAS_SOCIODEMO +
    CATEGORICAS_DIGITAL +
    CATEGORICAS_CAPACIDAD
)


# Score y política de crédito
SCORE_MIN = 0
SCORE_MAX = 1000

LGD = 0.50  # para la pérdida esperada EL = PD * EAD * LGD

# Score = 1000 * (1 - PD). Aprueba si PD <= 0.30, revisión manual hasta 0.45, rechaza por encima.
UMBRAL_APROBACION = 0.30
UMBRAL_REVISION = 0.45

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

# Hiperparámetros base. Los optimizados salen de busqueda_hiperparametros.py (models/best_params.json).
PARAMS_BASE_XGBOOST = {
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
    "random_state": SEMILLA,
    "n_jobs": -1,
}

PARAMS_BASE_RF = {
    "n_estimators": 150,
    "max_depth": 5,
    "min_samples_split": 5,
    "class_weight": "balanced",
    "random_state": SEMILLA,
    "n_jobs": -1,
}

PARAMS_BASE_LOGREG = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": SEMILLA,
}

# Búsqueda con Optuna
FOLDS_CV = 5
INTENTOS_OPTUNA = {"XGBoost": 100, "Random_Forest": 60, "Regresion_Logistica": 30}
