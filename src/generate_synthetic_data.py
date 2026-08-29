"""
==============================================================================
PROYECTO DE TESIS - UNIVERSIDAD DE LIMA
Carrera: Ingeniería de Sistemas
Título: Modelo de evaluación de riesgo crediticio para microfinanzas e
        inclusión financiera mediante XGBoost e Inteligencia Artificial
        Explicable (SHAP)
Archivo: src/generate_synthetic_data.py
Descripción: Generador de conjunto de datos sintético realista para microfinanzas
             en el Perú (Mypes, independientes, segmentos thin-file / no bancarizados).
==============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Importar configuración
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config


def generate_microfinance_dataset(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Genera un conjunto de datos representativo de solicitudes de microcrédito
    en el sistema microfinanciero peruano (Cajas Municipales, Financieras, Edpymes).

    Incorpora variables tradicionales y fuentes de datos alternativos (thin-file)
    para evaluar inclusión financiera (ODS 8 y 10).

    Args:
        n_samples (int): Número de solicitudes a simular (default 10,000).
        random_state (int): Semilla pseudoaleatoria para reproducibilidad.

    Returns:
        pd.DataFrame: DataFrame con las características y la etiqueta default_90d.
    """
    np.random.seed(random_state)

    # 1. Identificador de Solicitud
    ids = [f"CLI-{10000 + i}" for i in range(n_samples)]

    # 2. Variables Socio-Demográficas
    edad = np.random.randint(20, 68, size=n_samples)
    genero = np.random.choice(["F", "M"], size=n_samples, p=[0.52, 0.48]) # Alta participación femenina en microfinanzas
    estado_civil = np.random.choice(
        config.ESTADOS_CIVILES, size=n_samples, p=[0.30, 0.35, 0.25, 0.07, 0.03]
    )
    nivel_educacion = np.random.choice(
        config.NIVELES_EDUCACION, size=n_samples, p=[0.15, 0.45, 0.20, 0.08, 0.10, 0.02]
    )
    tipo_vivienda = np.random.choice(
        config.TIPOS_VIVIENDA, size=n_samples, p=[0.45, 0.30, 0.20, 0.05]
    )
    region_peru = np.random.choice(
        config.REGIONES_PERU, size=n_samples,
        p=[0.40, 0.10, 0.08, 0.08, 0.08, 0.06, 0.05, 0.05, 0.04, 0.03, 0.03]
    )

    # 3. Variables de Negocio y Operativas
    rubro_negocio = np.random.choice(
        config.RUBROS_NEGOCIO, size=n_samples,
        p=[0.35, 0.18, 0.12, 0.12, 0.10, 0.08, 0.05]
    )
    
    # Antigüedad del negocio en meses (distribución gamma sesgada a la derecha)
    antiguedad_negocio_meses = np.clip(
        np.random.gamma(shape=3.0, scale=12.0, size=n_samples).astype(int) + 3,
        3, 240
    )

    # Informalidad del negocio (1: Sin RUC o RUC 10 informal, 0: Formalizado)
    prob_informal = 0.65 - 0.0015 * antiguedad_negocio_meses
    prob_informal = np.clip(prob_informal, 0.25, 0.85)
    es_informal = (np.random.rand(n_samples) < prob_informal).astype(int)

    # 4. Variables Financieras (Moneda: Soles PEN S/)
    # Ingreso mensual neto generado por el negocio / solicitante (Distribución Log-Normal)
    ingreso_mensual_neto = np.round(
        np.exp(np.random.normal(loc=7.8, scale=0.55, size=n_samples)), -1
    )
    ingreso_mensual_neto = np.clip(ingreso_mensual_neto, 1100.0, 35000.0)

    # Gasto operativo mensual del negocio (entre 45% y 75% del ingreso)
    ratio_gasto_base = np.random.uniform(0.45, 0.75, size=n_samples)
    gasto_mensual_operativo = np.round(ingreso_mensual_neto * ratio_gasto_base, -1)

    # Monto solicitado de microcrédito (entre S/ 1,000 y S/ 45,000)
    monto_solicitado = np.round(
        np.clip(
            ingreso_mensual_neto * np.random.uniform(1.2, 3.8, size=n_samples),
            1000.0, 45000.0
        ), -2
    )

    # Plazo solicitado en meses (6, 12, 18, 24, 36, 48)
    plazos_posibles = np.array([6, 12, 18, 24, 36, 48])
    plazo_meses = np.random.choice(plazos_posibles, size=n_samples, p=[0.10, 0.35, 0.25, 0.20, 0.07, 0.03])

    # Tasa de Interés Efectiva Anual (TEA en microfinanzas en Perú: 25% a 65%)
    tasa_interes_anual = np.round(
        np.random.uniform(0.28, 0.58, size=n_samples) - 0.0005 * antiguedad_negocio_meses,
        3
    )
    tasa_interes_anual = np.clip(tasa_interes_anual, 0.22, 0.65)

    # Consultas a centrales de riesgo (SBS/Infocorp) en los últimos 6 meses
    num_consultas_centrales_ult_6m = np.random.poisson(lam=1.6, size=n_samples)
    num_consultas_centrales_ult_6m = np.clip(num_consultas_centrales_ult_6m, 0, 10)

    # Garantía o aval prendario / solidario
    posee_garantia_o_aval = np.random.choice([0, 1], size=n_samples, p=[0.60, 0.40])

    # 5. Variables Alternativas / Thin-File (Inclusión Financiera - ODS 10)
    # Puntualidad de pago de servicios públicos (Luz del Sur, Enel, Sedapal, etc. en los últimos 12 meses)
    puntualidad_servicios_basicos = np.round(
        np.clip(np.random.beta(a=6.0, b=2.5, size=n_samples), 0.0, 1.0), 2
    )

    # Uso de billeteras digitales (Yape, Plin) - Transacciones mensuales de cobro
    frecuencia_transacciones_yape_mes = np.random.poisson(lam=35.0, size=n_samples)
    frecuencia_transacciones_yape_mes = np.clip(frecuencia_transacciones_yape_mes, 0, 180)

    volumen_transacciones_digitales = np.round(
        frecuencia_transacciones_yape_mes * np.random.uniform(25.0, 95.0, size=n_samples), -1
    )

    # Antigüedad de la línea móvil celular (meses continuos sin cambio de número)
    antiguedad_celular_meses = np.clip(
        np.random.gamma(shape=2.5, scale=14.0, size=n_samples).astype(int) + 2,
        2, 120
    )

    # Variabilidad de ingresos / estacionalidad del flujo de ventas (Coeficiente de Variación)
    variabilidad_ingresos_cv = np.round(
        np.clip(np.random.beta(a=2.5, b=5.0, size=n_samples), 0.05, 0.70), 3
    )

    # 6. Modelado de la Probabilidad Real de Incumplimiento (Ground Truth Latente)
    # Calculamos cuota estimada para incluirla en la función de riesgo latente
    tasa_mensual = (1.0 + tasa_interes_anual) ** (1.0 / 12.0) - 1.0
    cuota_estimada = monto_solicitado * (
        tasa_mensual * (1.0 + tasa_mensual) ** plazo_meses
    ) / ((1.0 + tasa_mensual) ** plazo_meses - 1.0)

    flujo_disponible = ingreso_mensual_neto - gasto_mensual_operativo
    dti = cuota_estimada / ingreso_mensual_neto
    dscr = np.clip(flujo_disponible / (cuota_estimada + 1e-6), 0.1, 15.0)

    rubro_mayor_riesgo = np.isin(
        rubro_negocio, ["Transporte / Mototaxi / Taxi", "Gastronomía / Restaurante"]
    ).astype(float) * 0.35

    # Log-Odds de Default calibrado a ~12%
    log_odds = (
        -1.45                                               # Intercepto calibrado
        + 4.2 * (dti - 0.25)                               # Mayor endeudamiento eleva riesgo
        - 0.35 * np.log1p(dscr)                            # Mayor cobertura reduce riesgo
        - 0.008 * antiguedad_negocio_meses                 # Negocios más antiguos son más estables
        - 2.2 * (puntualidad_servicios_basicos - 0.70)     # Pagos puntuales de servicios reducen riesgo
        - 0.00018 * volumen_transacciones_digitales        # Actividad digital constante reduce riesgo
        - 0.006 * antiguedad_celular_meses                 # Estabilidad telefónica reduce riesgo
        + 2.1 * (variabilidad_ingresos_cv - 0.25)          # Alta variabilidad/estacionalidad eleva riesgo
        + 0.22 * num_consultas_centrales_ult_6m            # Múltiples consultas recientes indican sobreendeudamiento
        + 0.40 * es_informal                               # Informalidad añade leve prima de riesgo
        - 0.45 * posee_garantia_o_aval                     # Garantía mitiga el riesgo
        + rubro_mayor_riesgo
    )

    prob_default = 1.0 / (1.0 + np.exp(-log_odds))
    prob_default = np.clip(prob_default, 0.01, 0.95)

    # Generación de la etiqueta binaria de incumplimiento (Default a 90 días)
    default_90d = (np.random.rand(n_samples) < prob_default).astype(int)

    # Consolidación del DataFrame
    df = pd.DataFrame({
        "id_solicitante": ids,
        "edad": edad,
        "genero": genero,
        "estado_civil": estado_civil,
        "nivel_educacion": nivel_educacion,
        "tipo_vivienda": tipo_vivienda,
        "region_peru": region_peru,
        "rubro_negocio": rubro_negocio,
        "antiguedad_negocio_meses": antiguedad_negocio_meses,
        "es_informal": es_informal,
        "ingreso_mensual_neto": ingreso_mensual_neto,
        "gasto_mensual_operativo": gasto_mensual_operativo,
        "monto_solicitado": monto_solicitado,
        "plazo_meses": plazo_meses,
        "tasa_interes_anual": tasa_interes_anual,
        "num_consultas_centrales_ult_6m": num_consultas_centrales_ult_6m,
        "posee_garantia_o_aval": posee_garantia_o_aval,
        "puntualidad_servicios_basicos": puntualidad_servicios_basicos,
        "volumen_transacciones_digitales": volumen_transacciones_digitales,
        "frecuencia_transacciones_yape_mes": frecuencia_transacciones_yape_mes,
        "antiguedad_celular_meses": antiguedad_celular_meses,
        "variabilidad_ingresos_cv": variabilidad_ingresos_cv,
        "default_90d": default_90d
    })

    return df


def main():
    """Función de ejecución para generar y persistir el dataset raw."""
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_microfinance_dataset(n_samples=10000, random_state=config.RANDOM_STATE)
    df.to_csv(config.RAW_DATASET_PATH, index=False)
    
    tasa_default = df[config.TARGET_COLUMN].mean() * 100
    print(f"[OK] Dataset generado exitosamente en: {config.RAW_DATASET_PATH}")
    print(f"     Total registros: {len(df):,}")
    print(f"     Tasa de incumplimiento (Default rate): {tasa_default:.2f}%")
    print(f"     Total columnas: {df.shape[1]}")


if __name__ == "__main__":
    main()
