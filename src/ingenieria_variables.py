import numpy as np
import pandas as pd
from pathlib import Path

from src import config


def calcular_indice_inclusion_digital(
    adopcion_yape: np.ndarray,
    frecuencia_yape: np.ndarray,
    pago_celular: np.ndarray,
    uso_whatsapp: np.ndarray,
    uso_internet: np.ndarray
) -> np.ndarray:
    """Índice de inclusión digital (0 a 1): uso de billeteras, celular, WhatsApp e internet."""
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


def calcular_indice_resiliencia(
    afronta_imprevisto: np.ndarray,
    meses_desempleo: np.ndarray,
    sobra_fin_mes: np.ndarray,
    dificultad_1000: np.ndarray
) -> np.ndarray:
    """Índice de resiliencia (0 a 1): qué tan bien aguanta un imprevisto o perder su ingreso."""
    imprev_norm = np.where(afronta_imprevisto == "Sí", 1.0, 0.0)
    desemp_norm = np.clip((meses_desempleo - 1.0) / 4.0, 0.0, 1.0)
    sobra_norm = np.clip((sobra_fin_mes - 1.0) / 4.0, 0.0, 1.0)
    # dificultad_1000 va de 1 (nada difícil) a 4, por eso se invierte
    facilidad_1000 = np.clip((4.0 - dificultad_1000) / 3.0, 0.0, 1.0)

    indice = (
        0.35 * imprev_norm +
        0.25 * desemp_norm +
        0.20 * sobra_norm +
        0.20 * facilidad_1000
    )
    return np.round(np.clip(indice, 0.0, 1.0), 4)


def calcular_indice_vulnerabilidad(
    apenas_alcanza: np.ndarray,
    sobreendeudado: np.ndarray,
    credito_negado: np.ndarray
) -> np.ndarray:
    """Índice de vulnerabilidad de liquidez (0 a 1): ingreso que no alcanza, deudas, crédito negado."""
    alcanza_norm = np.clip((apenas_alcanza - 1.0) / 4.0, 0.0, 1.0)
    deuda_norm = np.clip((sobreendeudado - 1.0) / 4.0, 0.0, 1.0)
    negado_norm = np.where(credito_negado == "Sí", 1.0, 0.0)

    indice = 0.45 * alcanza_norm + 0.40 * deuda_norm + 0.15 * negado_norm
    return np.round(np.clip(indice, 0.0, 1.0), 4)


def calcular_indice_cultura_pago(
    puntualidad_deudas: np.ndarray,
    guarda_recibos: np.ndarray,
    anota_recibos: np.ndarray
) -> np.ndarray:
    """Índice de cultura de pago (0 a 1): puntualidad y orden con los recibos."""
    punt_norm = np.clip((puntualidad_deudas - 1.0) / 4.0, 0.0, 1.0)
    guarda_norm = np.where(guarda_recibos == "Sí", 1.0, 0.0)
    anota_norm = np.where(anota_recibos == "Sí", 1.0, 0.0)

    indice = 0.50 * punt_norm + 0.25 * guarda_norm + 0.25 * anota_norm
    return np.round(np.clip(indice, 0.0, 1.0), 4)


def agregar_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega los 4 índices a un DataFrame con las variables del cuestionario."""
    df_salida = df.copy()

    df_salida["indice_inclusion_digital"] = calcular_indice_inclusion_digital(
        df_salida["adopcion_yape_plin"].values,
        df_salida["frecuencia_yape_plin"].values.astype(float),
        df_salida["pago_celular_frecuencia"].values.astype(float),
        df_salida["uso_whatsapp"].values.astype(float),
        df_salida["uso_internet"].values.astype(float),
    )

    df_salida["indice_resiliencia_financiera"] = calcular_indice_resiliencia(
        df_salida["capacidad_afrontar_imprevisto"].values,
        df_salida["meses_resiliencia_desempleo"].values.astype(float),
        df_salida["sobra_dinero_fin_mes"].values.astype(float),
        df_salida["dificultad_emergencia_1000"].values.astype(float),
    )

    df_salida["indice_vulnerabilidad_liquidez"] = calcular_indice_vulnerabilidad(
        df_salida["ingreso_apenas_alcanza"].values.astype(float),
        df_salida["percepcion_sobreendeudado"].values.astype(float),
        df_salida["credito_formal_negado"].values,
    )

    df_salida["indice_cultura_pago"] = calcular_indice_cultura_pago(
        df_salida["puntualidad_pago_deudas"].values.astype(float),
        df_salida["guarda_dinero_recibos"].values,
        df_salida["anota_recibos_vencer"].values,
    )

    return df_salida


def cargar_encuesta(ruta_dta: Path = None) -> pd.DataFrame:
    """Lee el .dta de la encuesta SBS/CAF 2022 y arma una columna por variable del modelo."""
    ruta = ruta_dta or config.RUTA_DTA
    if not Path(ruta).exists():
        raise FileNotFoundError(f"No se encontró el archivo Stata en: {ruta}")

    df_crudo = pd.read_stata(ruta, convert_categoricals=False)

    df = pd.DataFrame()

    # Objetivo. F12_16_Rp: "pagué mis cuentas tarde; fallé en los pagos".
    # Quien no tuvo déficit (F11 == 1) no responde F12, queda en NaN y cuenta como 0.
    f12_16 = df_crudo["F12_16_Rp"].fillna(2.0)
    df[config.COLUMNA_OBJETIVO] = (f12_16 == 1.0).astype(int)

    # Eje 1: sociodemográfico. 97/98/99 son "no sabe / no responde".
    edad = df_crudo["D01_1"].replace({97: np.nan, 98: np.nan, 99: np.nan})
    df["edad"] = edad.fillna(edad.median()).astype(float)

    adultos = df_crudo["D05_AD"].replace({97: np.nan, 98: np.nan, 99: np.nan}).fillna(2.0)
    ninos = df_crudo["D05_CH"].replace({97: np.nan, 98: np.nan, 99: np.nan}).fillna(1.0)
    df["carga_familiar_adultos"] = adultos.astype(float)
    df["carga_familiar_ninos"] = ninos.astype(float)

    # D01: 1 hombre, 2 mujer
    df["genero"] = df_crudo["D01"].map({1.0: "Hombre", 2.0: "Mujer"}).fillna("Mujer")

    def mapear_educacion(v):
        if v in [1.0, 2.0]:
            return "Primaria / Sin Instrucción"
        elif v == 3.0:
            return "Secundaria Completa"
        elif v == 4.0:
            return "Superior Técnico Completo"
        elif v in [5.0, 6.0]:
            return "Superior Universitario Completo"
        return "Secundaria Completa"

    df["nivel_educacion"] = df_crudo["D09"].apply(mapear_educacion)

    # D02; Lima (14) y Callao (25) van juntos
    mapa_departamentos = {
        1.0: "Amazonas", 2.0: "Áncash", 3.0: "Apurímac", 4.0: "Arequipa", 5.0: "Ayacucho",
        6.0: "Cajamarca", 7.0: "Cusco", 8.0: "Huancavelica", 9.0: "Huánuco", 10.0: "Ica",
        11.0: "Junín", 12.0: "La Libertad", 13.0: "Lambayeque", 14.0: "Lima / Callao",
        15.0: "Loreto", 16.0: "Madre de Dios", 17.0: "Moquegua", 18.0: "Pasco",
        19.0: "Piura", 20.0: "Puno", 21.0: "San Martín", 22.0: "Tacna", 23.0: "Tumbes",
        24.0: "Ucayali", 25.0: "Lima / Callao"
    }
    df["departamento"] = df_crudo["D02"].map(mapa_departamentos).fillna("Lima / Callao")

    df["ambito"] = df_crudo["PA"].map({1.0: "Urbano", 2.0: "Rural"}).fillna("Urbano")

    def mapear_condicion_laboral(v):
        if v == 2.0:
            return "Trabajador Independiente / Cuenta propia"
        elif v == 3.0:
            return "Empleado Técnico o Profesional"
        elif v in [4.0, 5.0, 6.0]:
            return "Obrero / Empleado Operativo"
        elif v == 1.0:
            return "Empleador o Patrono (con personal)"
        elif v == 8.0:
            return "Campesino / Agricultor / Ganadero"
        return "Trabajador Independiente / Cuenta propia"

    df["condicion_laboral"] = df_crudo["D10"].apply(mapear_condicion_laboral)

    # Informal si nunca emite con RUC, no tiene beneficios de ley o es independiente
    informal = (
        (df_crudo["D05A"] == 1.0) |
        (df_crudo["D05B"] == 2.0) |
        (df_crudo["D10"] == 2.0)
    )
    df["es_informal"] = np.where(informal, "Informal / Sin RUC", "Formalizado / Con RUC o Beneficios")

    # Eje 2: inclusión digital. P0105: conoce o usa Yape/Plin/BIM
    df["adopcion_yape_plin"] = np.where(df_crudo["P0105"] == 1.0, "Sí", "No")

    # P0205: la tiene activa
    df["tiene_billetera_activa"] = np.where(df_crudo["P0205"] == 1.0, "Sí", "No")

    # P09_6_Rp: frecuencia de uso, 1 nunca a 4 siempre
    frec_yape = df_crudo["P09_6_Rp"].replace({97: 1, 98: 1, 99: 1}).fillna(1.0)
    df["frecuencia_yape_plin"] = frec_yape.astype(float)

    # S01_5_Rp: cobra/paga con el celular (1 a 5)
    pago_cel = df_crudo["S01_5_Rp"].replace({97: 2, 98: 2, 99: 2}).fillna(2.0)
    df["pago_celular_frecuencia"] = pago_cel.astype(float)

    wsp = df_crudo["D06_6_Rp"].replace({97: 1, 98: 1, 99: 1}).fillna(2.0)
    df["uso_whatsapp"] = wsp.astype(float)

    net = df_crudo["D06_13_Rp"].replace({97: 1, 98: 1, 99: 1}).fillna(2.0)
    df["uso_internet"] = net.astype(float)

    # S01_10_Rp: "el efectivo es más seguro" (1 a 5)
    pref_efectivo = df_crudo["S01_10_Rp"].replace({97: 3, 98: 3, 99: 3}).fillna(3.0)
    df["preferencia_efectivo"] = pref_efectivo.astype(float)

    # S04_4_Rp: "las herramientas digitales me facilitan" (1 a 5)
    conf_digital = df_crudo["S04_4_Rp"].replace({97: 3, 98: 3, 99: 3}).fillna(3.0)
    df["confianza_herramientas_digitales"] = conf_digital.astype(float)

    # F02_3_Rp y F02_4_Rp: guarda plata para recibos / anota vencimientos
    df["guarda_dinero_recibos"] = np.where(df_crudo["F02_3_Rp"] == 1.0, "Sí", "No")

    df["anota_recibos_vencer"] = np.where(df_crudo["F02_4_Rp"] == 1.0, "Sí", "No")

    # S02_5_Rp: paga a tiempo, 1 nunca a 5 siempre
    punt_deudas = df_crudo["S02_5_Rp"].replace({97: 3, 98: 3, 99: 3}).fillna(4.0)
    df["puntualidad_pago_deudas"] = punt_deudas.astype(float)

    # Eje 3: capacidad financiera. D13A: rango de ingreso
    def mapear_rango_ingreso(v):
        if v in [1.0, 2.0]:
            return "Menor a S/ 600"
        elif v == 3.0:
            return "De S/ 601 a S/ 1,200 (Ingreso Mínimo)"
        elif v == 4.0:
            return "De S/ 1,201 a S/ 2,400"
        elif v in [5.0, 6.0]:
            return "De S/ 2,401 a S/ 4,000"
        elif v in [7.0, 8.0]:
            return "Mayor a S/ 4,000"
        return "De S/ 601 a S/ 1,200 (Ingreso Mínimo)"

    df["rango_ingreso"] = df_crudo["D13A"].apply(mapear_rango_ingreso)

    # NSEX: 1 A ... 5 E (A y B se juntan)
    def mapear_nse(v):
        if v in [1.0, 2.0]:
            return "A/B"
        elif v == 3.0:
            return "C"
        elif v == 4.0:
            return "D"
        elif v == 5.0:
            return "E"
        return "C"

    df["nivel_socioeconomico"] = df_crudo["NSEX"].apply(mapear_nse)

    # D13: ingresos regulares
    df["ingresos_estables"] = np.where(df_crudo["D13"] == 1.0, "Sí", "No")

    df["planifica_presupuesto"] = np.where(df_crudo["F02_1_Rp"] == 1.0, "Sí", "No")
    df["anota_gastos"] = np.where(df_crudo["F02_2_Rp"] == 1.0, "Sí", "No")

    # F04: puede cubrir un imprevisto sin endeudarse
    df["capacidad_afrontar_imprevisto"] = np.where(df_crudo["F04"] == 1.0, "Sí", "No")

    # C03: dificultad de juntar S/ 1,000, 1 nada difícil a 4
    dif_1000 = df_crudo["C03"].replace({97: 3, 98: 3, 99: 3}).fillna(3.0)
    df["dificultad_emergencia_1000"] = dif_1000.astype(float)

    # F13: cuánto aguanta si pierde su ingreso (1 a 5)
    desempleo_meses = df_crudo["F13"].replace({97: 2, 98: 2, 99: 2}).fillna(2.0)
    df["meses_resiliencia_desempleo"] = desempleo_meses.astype(float)

    sobra_dinero = df_crudo["S02_4_Rp"].replace({97: 2, 98: 2, 99: 2}).fillna(2.0)
    df["sobra_dinero_fin_mes"] = sobra_dinero.astype(float)

    apenas_alcanza = df_crudo["S03_6_Rp"].replace({97: 3, 98: 3, 99: 3}).fillna(3.0)
    df["ingreso_apenas_alcanza"] = apenas_alcanza.astype(float)

    sobreendeudado = df_crudo["S01_8_Rp"].replace({97: 2, 98: 2, 99: 2}).fillna(2.0)
    df["percepcion_sobreendeudado"] = sobreendeudado.astype(float)

    # Productos que tiene o tuvo
    df["tiene_cuenta_ahorro"] = np.where(df_crudo["P0101"] == 1.0, "Sí", "No")
    df["tiene_tarjeta_credito"] = np.where(df_crudo["P0109"] == 1.0, "Sí", "No")
    df["tiene_prestamo_personal"] = np.where(df_crudo["P0107"] == 1.0, "Sí", "No")
    df["ha_usado_prestamista"] = np.where(df_crudo["P0130"] == 1.0, "Sí", "No")
    df["ha_usado_pandero_junta"] = np.where(df_crudo["P0129"] == 1.0, "Sí", "No")
    df["credito_formal_negado"] = np.where(df_crudo["P10_9_Rp"] == 1.0, "Sí", "No")

    return agregar_indices(df)


if __name__ == "__main__":
    df_encuesta = cargar_encuesta()
    print(f"{df_encuesta.shape[0]} filas, {df_encuesta.shape[1]} columnas")
    print(f"mora: {df_encuesta[config.COLUMNA_OBJETIVO].mean()*100:.2f}%")
