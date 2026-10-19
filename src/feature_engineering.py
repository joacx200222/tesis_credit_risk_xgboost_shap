import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Importar configuración
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config


def calculate_digital_inclusion_index(
    adopcion_yape: np.ndarray,
    frecuencia_yape: np.ndarray,
    pago_celular: np.ndarray,
    uso_whatsapp: np.ndarray,
    uso_internet: np.ndarray
) -> np.ndarray:
    """
    Calcula el Índice de Inclusión y Transaccionalidad Digital (0.0 a 1.0).
    A mayor uso regular de billeteras móviles y herramientas digitales,
    mayor huella transaccional alternativa.
    """
    adop_norm = np.where(adopcion_yape == "Sí", 1.0, 0.0)
    frec_norm = np.clip((frecuencia_yape - 1.0) / 3.0, 0.0, 1.0)
    cel_norm = np.clip((pago_celular - 1.0) / 4.0, 0.0, 1.0)
    wsp_norm = np.clip((uso_whatsapp - 1.0) / 3.0, 0.0, 1.0)
    net_norm = np.clip((uso_internet - 1.0) / 3.0, 0.0, 1.0)

    indice = (
        0.30 * adop_norm +
        0.30 * frec_norm +
        0.20 * cel_norm +
        0.10 * wsp_norm +
        0.10 * net_norm
    )
    return np.round(np.clip(indice, 0.0, 1.0), 4)


def calculate_financial_resilience_index(
    afronta_imprevisto: np.ndarray,
    meses_desempleo: np.ndarray,
    sobra_fin_mes: np.ndarray,
    dificultad_1000: np.ndarray
) -> np.ndarray:
    """
    Calcula el Índice de Resiliencia Financiera (0.0 a 1.0).
    Mide la capacidad de absorber choques de liquidez sin recurrir al impago.
    """
    imprev_norm = np.where(afronta_imprevisto == "Sí", 1.0, 0.0)
    desemp_norm = np.clip((meses_desempleo - 1.0) / 4.0, 0.0, 1.0)
    sobra_norm = np.clip((sobra_fin_mes - 1.0) / 4.0, 0.0, 1.0)
    # dificultad_1000: 1: Nada difícil (alta resiliencia) a 4: Muy difícil
    facilidad_1000 = np.clip((4.0 - dificultad_1000) / 3.0, 0.0, 1.0)

    indice = (
        0.35 * imprev_norm +
        0.25 * desemp_norm +
        0.20 * sobra_norm +
        0.20 * facilidad_1000
    )
    return np.round(np.clip(indice, 0.0, 1.0), 4)


def calculate_liquidity_vulnerability_index(
    apenas_alcanza: np.ndarray,
    sobreendeudado: np.ndarray,
    credito_negado: np.ndarray
) -> np.ndarray:
    """
    Calcula el Índice de Vulnerabilidad de Liquidez (0.0 a 1.0).
    Captura el estrés financiero del hogar y la sobrecarga de obligaciones.
    """
    alcanza_norm = np.clip((apenas_alcanza - 1.0) / 4.0, 0.0, 1.0)
    deuda_norm = np.clip((sobreendeudado - 1.0) / 4.0, 0.0, 1.0)
    negado_norm = np.where(credito_negado == "Sí", 1.0, 0.0)

    indice = 0.45 * alcanza_norm + 0.40 * deuda_norm + 0.15 * negado_norm
    return np.round(np.clip(indice, 0.0, 1.0), 4)


def calculate_payment_culture_index(
    puntualidad_deudas: np.ndarray,
    guarda_recibos: np.ndarray,
    anota_recibos: np.ndarray
) -> np.ndarray:
    """
    Calcula el Índice de Cultura de Pago y Cumplimiento (0.0 a 1.0).
    Basado en hábitos de reserva y puntualidad en servicios esenciales.
    """
    punt_norm = np.clip((puntualidad_deudas - 1.0) / 4.0, 0.0, 1.0)
    guarda_norm = np.where(guarda_recibos == "Sí", 1.0, 0.0)
    anota_norm = np.where(anota_recibos == "Sí", 1.0, 0.0)

    indice = 0.50 * punt_norm + 0.25 * guarda_norm + 0.25 * anota_norm
    return np.round(np.clip(indice, 0.0, 1.0), 4)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el cálculo de los 4 índices sintéticos a cualquier DataFrame
    que contenga las variables del cuestionario.
    """
    df_out = df.copy()

    # 1. Índice de Inclusión Digital
    df_out["indice_inclusion_digital"] = calculate_digital_inclusion_index(
        df_out["adopcion_yape_plin"].values,
        df_out["frecuencia_yape_plin"].values.astype(float),
        df_out["pago_celular_frecuencia"].values.astype(float),
        df_out["uso_whatsapp"].values.astype(float),
        df_out["uso_internet"].values.astype(float),
    )

    # 2. Índice de Resiliencia Financiera
    df_out["indice_resiliencia_financiera"] = calculate_financial_resilience_index(
        df_out["capacidad_afrontar_imprevisto"].values,
        df_out["meses_resiliencia_desempleo"].values.astype(float),
        df_out["sobra_dinero_fin_mes"].values.astype(float),
        df_out["dificultad_emergencia_1000"].values.astype(float),
    )

    # 3. Índice de Vulnerabilidad de Liquidez
    df_out["indice_vulnerabilidad_liquidez"] = calculate_liquidity_vulnerability_index(
        df_out["ingreso_apenas_alcanza"].values.astype(float),
        df_out["percepcion_sobreendeudado"].values.astype(float),
        df_out["credito_formal_negado"].values,
    )

    # 4. Índice de Cultura de Pago
    df_out["indice_cultura_pago"] = calculate_payment_culture_index(
        df_out["puntualidad_pago_deudas"].values.astype(float),
        df_out["guarda_dinero_recibos"].values,
        df_out["anota_recibos_vencer"].values,
    )

    return df_out


def load_and_extract_sbs_dataset(dta_path: Path = None) -> pd.DataFrame:
    """
    Lee la base de datos empírica oficial SBS/CAF 2022 y mapea las preguntas
    del cuestionario a las columnas estandarizadas del modelo de scoring.
    """
    path = dta_path or config.RAW_DTA_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"No se encontró el archivo Stata en: {path}")

    # Cargar base cruda en Stata
    df_raw = pd.read_stata(path, convert_categoricals=False)

    df = pd.DataFrame()

    # VARIABLE OBJETIVO (TARGET: Default / Mora en Cuentas y Deudas)
    # F12_16_Rp: "p) Pagué mis cuentas tarde; fallé en los pagos"
    # Quienes no tuvieron déficit (F11 == 1) tienen NaN en F12 y son 0 (No default)
    f12_16_clean = df_raw["F12_16_Rp"].fillna(2.0)
    df[config.TARGET_COLUMN] = (f12_16_clean == 1.0).astype(int)

    # EJE 1: VARIABLES SOCIODEMOGRÁFICAS Y OCUPACIÓN
    # Edad
    edad = df_raw["D01_1"].replace({97: np.nan, 98: np.nan, 99: np.nan})
    df["edad"] = edad.fillna(edad.median()).astype(float)

    # Cargas familiares
    adultos = df_raw["D05_AD"].replace({97: np.nan, 98: np.nan, 99: np.nan}).fillna(2.0)
    ninos = df_raw["D05_CH"].replace({97: np.nan, 98: np.nan, 99: np.nan}).fillna(1.0)
    df["carga_familiar_adultos"] = adultos.astype(float)
    df["carga_familiar_ninos"] = ninos.astype(float)

    # Género: D01 (1: Hombre, 2: Mujer)
    df["genero"] = df_raw["D01"].map({1.0: "Hombre", 2.0: "Mujer"}).fillna("Mujer")

    # Nivel Educativo: D09
    def map_educacion(val):
        if val in [1.0, 2.0]:
            return "Primaria / Sin Instrucción"
        elif val == 3.0:
            return "Secundaria Completa"
        elif val == 4.0:
            return "Superior Técnico Completo"
        elif val in [5.0, 6.0]:
            return "Superior Universitario Completo"
        return "Secundaria Completa"

    df["nivel_educacion"] = df_raw["D09"].apply(map_educacion)

    # Departamento: D02
    dept_map = {
        1.0: "Amazonas", 2.0: "Áncash", 3.0: "Apurímac", 4.0: "Arequipa", 5.0: "Ayacucho",
        6.0: "Cajamarca", 7.0: "Cusco", 8.0: "Huancavelica", 9.0: "Huánuco", 10.0: "Ica",
        11.0: "Junín", 12.0: "La Libertad", 13.0: "Lambayeque", 14.0: "Lima / Callao",
        15.0: "Loreto", 16.0: "Madre de Dios", 17.0: "Moquegua", 18.0: "Pasco",
        19.0: "Piura", 20.0: "Puno", 21.0: "San Martín", 22.0: "Tacna", 23.0: "Tumbes",
        24.0: "Ucayali", 25.0: "Lima / Callao"
    }
    df["departamento"] = df_raw["D02"].map(dept_map).fillna("Lima / Callao")

    # Ámbito: PA (1: Urbano, 2: Rural)
    df["ambito"] = df_raw["PA"].map({1.0: "Urbano", 2.0: "Rural"}).fillna("Urbano")

    # Condición Laboral: D10
    def map_condicion_laboral(val):
        if val == 2.0:
            return "Trabajador Independiente / Cuenta propia"
        elif val == 3.0:
            return "Empleado Técnico o Profesional"
        elif val in [4.0, 5.0, 6.0]:
            return "Obrero / Empleado Operativo"
        elif val == 1.0:
            return "Empleador o Patrono (con personal)"
        elif val == 8.0:
            return "Campesino / Agricultor / Ganadero"
        return "Trabajador Independiente / Cuenta propia"

    df["condicion_laboral"] = df_raw["D10"].apply(map_condicion_laboral)

    # Informalidad laboral / Thin-file: D05A (RUC) y D05B (Beneficios)
    es_informal_flag = (
        (df_raw["D05A"] == 1.0) |  # Nunca emite recibo/factura con RUC
        (df_raw["D05B"] == 2.0) |  # No recibe beneficios de ley
        (df_raw["D10"] == 2.0)     # Independiente / cuenta propia
    )
    df["es_informal"] = np.where(es_informal_flag, "Informal / Sin RUC", "Formalizado / Con RUC o Beneficios")

    # EJE 2: INCLUSIÓN DIGITAL Y DATOS ALTERNATIVOS
    # P0105: Billetera móvil (Yape, Plin, BIM)
    df["adopcion_yape_plin"] = np.where(df_raw["P0105"] == 1.0, "Sí", "No")

    # P0205: Tenencia activa de billetera móvil
    df["tiene_billetera_activa"] = np.where(df_raw["P0205"] == 1.0, "Sí", "No")

    # P09_6_Rp: Frecuencia de uso billetera digital (1: Nunca a 4: Siempre)
    frec_yape = df_raw["P09_6_Rp"].replace({97: 1, 98: 1, 99: 1}).fillna(1.0)
    df["frecuencia_yape_plin"] = frec_yape.astype(float)

    # S01_5_Rp: Uso del celular para cobros y pagos (1 a 5)
    pago_cel = df_raw["S01_5_Rp"].replace({97: 2, 98: 2, 99: 2}).fillna(2.0)
    df["pago_celular_frecuencia"] = pago_cel.astype(float)

    # D06_6_Rp: Uso WhatsApp (1 a 4)
    wsp = df_raw["D06_6_Rp"].replace({97: 1, 98: 1, 99: 1}).fillna(2.0)
    df["uso_whatsapp"] = wsp.astype(float)

    # D06_13_Rp: Uso Internet (1 a 4)
    net = df_raw["D06_13_Rp"].replace({97: 1, 98: 1, 99: 1}).fillna(2.0)
    df["uso_internet"] = net.astype(float)

    # S01_10_Rp: Preferencia por efectivo vs medios digitales (1 a 5)
    pref_efectivo = df_raw["S01_10_Rp"].replace({97: 3, 98: 3, 99: 3}).fillna(3.0)
    df["preferencia_efectivo"] = pref_efectivo.astype(float)

    # S04_4_Rp: Herramientas digitales facilitan gestión (1 a 5)
    conf_digital = df_raw["S04_4_Rp"].replace({97: 3, 98: 3, 99: 3}).fillna(3.0)
    df["confianza_herramientas_digitales"] = conf_digital.astype(float)

    # Puntualidad de cuentas y recibos
    # F02_3_Rp: Guarda dinero para facturas
    df["guarda_dinero_recibos"] = np.where(df_raw["F02_3_Rp"] == 1.0, "Sí", "No")

    # F02_4_Rp: Anota recibos por vencer
    df["anota_recibos_vencer"] = np.where(df_raw["F02_4_Rp"] == 1.0, "Sí", "No")

    # S02_5_Rp: Pago a tiempo de cuentas y deudas (1: Nunca a 5: Siempre)
    punt_deudas = df_raw["S02_5_Rp"].replace({97: 3, 98: 3, 99: 3}).fillna(4.0)
    df["puntualidad_pago_deudas"] = punt_deudas.astype(float)

    # EJE 3: CAPACIDAD Y COMPORTAMIENTO FINANCIERO
    # Rango de Ingresos: D13A
    def map_rango_ingreso(val):
        if val in [1.0, 2.0]:
            return "Menor a S/ 600"
        elif val == 3.0:
            return "De S/ 601 a S/ 1,200 (Ingreso Mínimo)"
        elif val == 4.0:
            return "De S/ 1,201 a S/ 2,400"
        elif val in [5.0, 6.0]:
            return "De S/ 2,401 a S/ 4,000"
        elif val in [7.0, 8.0]:
            return "Mayor a S/ 4,000"
        return "De S/ 601 a S/ 1,200 (Ingreso Mínimo)"

    df["rango_ingreso"] = df_raw["D13A"].apply(map_rango_ingreso)

    # NSE: NSEX (1: A, 2: B, 3: C, 4: D, 5: E)
    def map_nse(val):
        if val in [1.0, 2.0]:
            return "A/B"
        elif val == 3.0:
            return "C"
        elif val == 4.0:
            return "D"
        elif val == 5.0:
            return "E"
        return "C"

    df["nivel_socioeconomico"] = df_raw["NSEX"].apply(map_nse)

    # Regularidad de ingresos: D13 (1: Sí es regular, 2: No)
    df["ingresos_estables"] = np.where(df_raw["D13"] == 1.0, "Sí", "No")

    # Presupuesto: F02_1_Rp y F02_2_Rp
    df["planifica_presupuesto"] = np.where(df_raw["F02_1_Rp"] == 1.0, "Sí", "No")
    df["anota_gastos"] = np.where(df_raw["F02_2_Rp"] == 1.0, "Sí", "No")

    # Resiliencia ante imprevistos: F04 (1: Sí sin endeudarse, 2: No)
    df["capacidad_afrontar_imprevisto"] = np.where(df_raw["F04"] == 1.0, "Sí", "No")

    # Dificultad para reunir S/ 1,000 de emergencia: C03 (1: Nada difícil a 4: Muy difícil)
    dif_1000 = df_raw["C03"].replace({97: 3, 98: 3, 99: 3}).fillna(3.0)
    df["dificultad_emergencia_1000"] = dif_1000.astype(float)

    # Meses de cobertura ante desempleo: F13 (1 a 5)
    desempleo_meses = df_raw["F13"].replace({97: 2, 98: 2, 99: 2}).fillna(2.0)
    df["meses_resiliencia_desempleo"] = desempleo_meses.astype(float)

    # Sobra dinero fin de mes: S02_4_Rp (1 a 5)
    sobra_dinero = df_raw["S02_4_Rp"].replace({97: 2, 98: 2, 99: 2}).fillna(2.0)
    df["sobra_dinero_fin_mes"] = sobra_dinero.astype(float)

    # Ingreso apenas alcanza: S03_6_Rp (1 a 5)
    apenas_alcanza = df_raw["S03_6_Rp"].replace({97: 3, 98: 3, 99: 3}).fillna(3.0)
    df["ingreso_apenas_alcanza"] = apenas_alcanza.astype(float)

    # Percepción de sobreendeudamiento: S01_8_Rp (1 a 5)
    sobreendeudado = df_raw["S01_8_Rp"].replace({97: 2, 98: 2, 99: 2}).fillna(2.0)
    df["percepcion_sobreendeudado"] = sobreendeudado.astype(float)

    # Historial de productos previos
    df["tiene_cuenta_ahorro"] = np.where(df_raw["P0101"] == 1.0, "Sí", "No")
    df["tiene_tarjeta_credito"] = np.where(df_raw["P0109"] == 1.0, "Sí", "No")
    df["tiene_prestamo_personal"] = np.where(df_raw["P0107"] == 1.0, "Sí", "No")
    df["ha_usado_prestamista"] = np.where(df_raw["P0130"] == 1.0, "Sí", "No")
    df["ha_usado_pandero_junta"] = np.where(df_raw["P0129"] == 1.0, "Sí", "No")
    df["credito_formal_negado"] = np.where(df_raw["P10_9_Rp"] == 1.0, "Sí", "No")

    # INGENIERÍA DE CARACTERÍSTICAS: CÁLCULO DE ÍNDICES SINTÉTICOS
    df_enriched = add_engineered_features(df)

    return df_enriched


if __name__ == "__main__":
    df_extracted = load_and_extract_sbs_dataset()
    print("[OK] Ingestión y feature engineering de la base SBS/CAF 2022 completada.")
    print(f"     Dimensiones resultantes: {df_extracted.shape}")
    print(f"     Tasa de morosidad (target): {df_extracted[config.TARGET_COLUMN].mean()*100:.2f}%")
    print(f"     Columnas extraídas: {list(df_extracted.columns)}")
