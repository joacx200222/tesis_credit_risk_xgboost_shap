"""
==============================================================================
PROYECTO DE TESIS - UNIVERSIDAD DE LIMA
Carrera: Ingeniería de Sistemas
Título: Modelo de evaluación de riesgo crediticio para microfinanzas e
        inclusión financiera mediante XGBoost e Inteligencia Artificial
        Explicable (SHAP)
Archivo: src/feature_engineering.py
Descripción: Transformaciones financieras, cálculo de cuotas y ratios de solvencia
             según estándares microfinancieros y de la SBS.
==============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Importar configuración
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config


def calculate_monthly_installment(monto: np.ndarray, tasa_anual: np.ndarray, plazo: np.ndarray) -> np.ndarray:
    """
    Calcula la cuota mensual constante mediante la fórmula de anualidad ordinaria (Método Francés):
        R = P * [ i*(1+i)^n ] / [ (1+i)^n - 1 ]
    donde i = (1 + TEA)^(1/12) - 1.

    Args:
        monto: Monto del préstamo (Principal P).
        tasa_anual: Tasa Efectiva Anual (TEA).
        plazo: Plazo en meses (n).

    Returns:
        np.ndarray: Vector de cuotas mensuales estimadas en Soles (PEN).
    """
    # Conversión de TEA a Tasa Efectiva Mensual (TEM)
    tasa_mensual = (1.0 + tasa_anual) ** (1.0 / 12.0) - 1.0
    
    # Manejo de casos límite
    numerador = monto * (tasa_mensual * (1.0 + tasa_mensual) ** plazo)
    denominador = ((1.0 + tasa_mensual) ** plazo) - 1.0
    
    cuota = np.where(denominador > 0, numerador / denominador, monto / plazo)
    return np.round(cuota, 2)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega al DataFrame todas las características derivadas financieras y
    de comportamiento alternativo necesarias para el modelo XGBoost.

    Args:
        df (pd.DataFrame): DataFrame con las características originales.

    Returns:
        pd.DataFrame: DataFrame enriquecido con las nuevas columnas calculadas.
    """
    df_out = df.copy()

    # 1. Cuota mensual calculada
    df_out["cuota_mensual"] = calculate_monthly_installment(
        df_out["monto_solicitado"].values,
        df_out["tasa_interes_anual"].values,
        df_out["plazo_meses"].values
    )

    # 2. Flujo de Caja Libre Mensual (EBITDA aproximado de la microempresa)
    df_out["flujo_caja_libre_mensual"] = (
        df_out["ingreso_mensual_neto"] - df_out["gasto_mensual_operativo"]
    )

    # 3. Ratio Cuota / Ingreso Neto (Debt-to-Income Ratio - DTI)
    # Criterio SBS: Ratios superiores a 0.35 - 0.40 representan alerta de sobreendeudamiento
    df_out["ratio_cuota_ingreso"] = np.round(
        df_out["cuota_mensual"] / (df_out["ingreso_mensual_neto"] + 1e-5), 4
    )

    # 4. Ratio de Cobertura del Servicio de la Deuda (Debt Service Coverage Ratio - DSCR)
    # DSCR = Flujo de Caja Disponible / Cuota Mensual. Valores < 1.2 indican vulnerabilidad
    df_out["ratio_cobertura_deuda"] = np.round(
        df_out["flujo_caja_libre_mensual"] / (df_out["cuota_mensual"] + 1e-5), 4
    )
    df_out["ratio_cobertura_deuda"] = np.clip(df_out["ratio_cobertura_deuda"], -5.0, 20.0)

    # 5. Ratio Monto Solicitado vs Ingreso Neto
    df_out["ratio_monto_ingreso"] = np.round(
        df_out["monto_solicitado"] / (df_out["ingreso_mensual_neto"] + 1e-5), 4
    )

    # 6. Índice de Estabilidad del Negocio (Combinación no lineal de Antigüedad y Estacionalidad)
    # A mayor antigüedad y menor coeficiente de variación de ingresos, mayor estabilidad
    antiguedad = df_out["antiguedad_negocio_meses"].values
    cv = df_out["variabilidad_ingresos_cv"].values
    df_out["indice_estabilidad_negocio"] = np.round(
        np.log1p(antiguedad) / (1.0 + 2.0 * cv), 4
    )

    # 7. Score Compuesto de Comportamiento Alternativo (Inclusión Financiera ODS 10)
    # Ponderación de puntualidad de servicios (45%), volumen digital (35%) y estabilidad de línea (20%)
    puntualidad = df_out["puntualidad_servicios_basicos"].values
    vol_norm = np.clip(df_out["volumen_transacciones_digitales"].values / 2500.0, 0.0, 1.0)
    cel_norm = np.clip(df_out["antiguedad_celular_meses"].values / 36.0, 0.0, 1.0)
    
    score_alt = 0.45 * puntualidad + 0.35 * vol_norm + 0.20 * cel_norm
    df_out["score_comportamiento_alternativo"] = np.round(score_alt, 4)

    return df_out


if __name__ == "__main__":
    # Prueba rápida del módulo
    df_raw = pd.read_csv(config.RAW_DATASET_PATH)
    df_feat = add_engineered_features(df_raw)
    print("[OK] Ingeniería de características ejecutada con éxito.")
    print("     Nuevas variables calculadas:")
    for col in config.NUMERICAL_DERIVED_FEATURES:
        print(f"       - {col}: min={df_feat[col].min():.2f}, max={df_feat[col].max():.2f}, media={df_feat[col].mean():.2f}")
