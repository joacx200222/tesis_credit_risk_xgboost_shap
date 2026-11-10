import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import shap

from src import config
from src.preprocesamiento import Preprocesador
from src.entrenamiento import ModeloXGBoost
from src.ingenieria_variables import agregar_indices
from src.evaluacion import (
    perdida_esperada,
    pd_a_score,
    decision_crediticia
)


NOMBRES_VARIABLES = {
    "indice_inclusion_digital": "Índice de Inclusión Digital (Yape/Plin)",
    "indice_resiliencia_financiera": "Resiliencia ante Gastos Imprevistos",
    "indice_vulnerabilidad_liquidez": "Vulnerabilidad Financiera Cotidiana",
    "indice_cultura_pago": "Cultura de Pago y Puntualidad en Cuentas",
    "frecuencia_yape_plin": "Frecuencia de Uso de Yape / Plin / BIM",
    "pago_celular_frecuencia": "Uso de Celular para Cobros/Pagos",
    "uso_whatsapp": "Frecuencia de Uso de WhatsApp",
    "uso_internet": "Frecuencia de Navegación en Internet",
    "preferencia_efectivo": "Preferencia por Efectivo vs. Digital",
    "confianza_herramientas_digitales": "Confianza en Gestión Financiera Digital",
    "puntualidad_pago_deudas": "Puntualidad en Pago de Obligaciones",
    "dificultad_emergencia_1000": "Dificultad de Reunir S/ 1,000 en Emergencia",
    "meses_resiliencia_desempleo": "Meses de Cobertura ante Desempleo",
    "sobra_dinero_fin_mes": "Margen de Dinero Sobrante a Fin de Mes",
    "ingreso_apenas_alcanza": "Ingreso Apenas Alcanza para Sobrevivir",
    "percepcion_sobreendeudado": "Percepción de Sobreendeudamiento",
    "edad": "Edad del Solicitante",
    "carga_familiar_adultos": "Adultos Dependientes en el Hogar",
    "carga_familiar_ninos": "Hijos Menores a Cargo",
    "adopcion_yape_plin_Sí": "Dispone de Billetera Móvil (Yape/Plin)",
    "adopcion_yape_plin_No": "No usa Billeteras Digitales",
    "tiene_billetera_activa_Sí": "Billetera Móvil Activa",
    "guarda_dinero_recibos_Sí": "Reserva Dinero para Servicios Básicos",
    "guarda_dinero_recibos_No": "No Reserva Dinero para Servicios",
    "anota_recibos_vencer_Sí": "Registra Vencimiento de Recibos",
    "anota_recibos_vencer_No": "No Lleva Control de Vencimientos",
    "planifica_presupuesto_Sí": "Planifica Presupuesto Mensual",
    "anota_gastos_Sí": "Registra sus Gastos Diarios",
    "ingresos_estables_Sí": "Ingresos Mensuales Regulares",
    "ingresos_estables_No": "Ingresos Irregulares / Inestables",
    "capacidad_afrontar_imprevisto_Sí": "Afronta Emergencias sin Endeudarse",
    "capacidad_afrontar_imprevisto_No": "Sin Ahorro para Imprevistos",
    "es_informal_Informal / Sin RUC": "Microempresario Informal / Sin RUC",
    "es_informal_Formalizado / Con RUC o Beneficios": "Negocio Formalizado",
    "condicion_laboral_Trabajador Independiente / Cuenta propia": "Trabajador Independiente",
    "ha_usado_prestamista_Sí": "Historial con Prestamistas Informales",
    "ha_usado_pandero_junta_Sí": "Participación en Juntas / Panderos",
    "credito_formal_negado_Sí": "Solicitud Crediticia Negada Previamente",
    "tiene_cuenta_ahorro_Sí": "Posee Cuenta de Ahorro Bancaria",
    "tiene_tarjeta_credito_Sí": "Posee Tarjeta de Crédito",
    "tiene_prestamo_personal_Sí": "Cuenta con Préstamo Personal",
    "ambito_Rural": "Zona Geográfica Rural",
    "ambito_Urbano": "Zona Geográfica Urbana",
    "genero_Mujer": "Género: Femenino",
    "genero_Hombre": "Género: Masculino"
}


# Texto que se le muestra al solicitante cuando esa variable empuja hacia la mora
MOTIVOS_DENEGACION = {
    "indice_vulnerabilidad_liquidez": "Insuficiencia habitual de ingresos familiares reportada en el hogar.",
    "indice_resiliencia_financiera": "Incapacidad para absorber gastos de emergencia sin recurrir a endeudamiento adicional.",
    "indice_inclusion_digital": "Falta de historial transaccional demostrable en medios de pago digitales (Yape/Plin).",
    "indice_cultura_pago": "Incumplimiento o retrasos frecuentes en el pago de recibos de servicios esenciales.",
    "percepcion_sobreendeudado": "Sobrecarga de pasivos preexistentes y percepción de sobreendeudamiento.",
    "ingreso_apenas_alcanza": "Estrechez en el flujo de caja del hogar respecto al costo de vida.",
    "puntualidad_pago_deudas": "Historial o hábito irregular en la puntualidad de obligaciones asumidas.",
    "dificultad_emergencia_1000": "Severa dificultad económica para consolidar un fondo de contingencia básico (S/ 1,000).",
    "ha_usado_prestamista_Sí": "Exposición crediticia previa en circuitos informales no regulados (prestamistas).",
    "credito_formal_negado_Sí": "Registro de rechazos previos en solicitudes de crédito ante entidades financieras supervisadas.",
    "guarda_dinero_recibos_No": "Ausencia de reservas preventivas para el pago de facturas y obligaciones corrientes.",
    "capacidad_afrontar_imprevisto_No": "Carencia de margen de maniobra financiero ante choques de liquidez inesperados.",
    "preferencia_efectivo": "Aversión marcada al uso de canales formales digitales, limitando la trazabilidad de ingresos.",
    "es_informal_Informal / Sin RUC": "Actividad económica no documentada formalmente ante la administración tributaria (RUC).",
    "ingresos_estables_No": "Alta variabilidad e inestabilidad temporal en los flujos periódicos de ingresos."
}


class ExplicadorSHAP:
    """TreeSHAP sobre el XGBoost: explicación local, importancia global y gráficos."""

    def __init__(
        self,
        modelo: ModeloXGBoost,
        preprocesador: Preprocesador
    ):
        self.modelo = modelo
        self.preprocesador = preprocesador
        self.nombres_variables = preprocesador.get_feature_names_out()

        nativo = modelo.xgb
        self.tree_explainer = shap.TreeExplainer(nativo)

        # Algunas versiones de shap con xgboost >= 2 leen mal el base_score y dan
        # expected_value = 0. Se despeja de f(x) = E[f(x)] + sum(phi) en un punto cualquiera.
        x_ref = np.zeros((1, len(self.nombres_variables)))
        margen_ref = float(nativo.predict(x_ref, output_margin=True)[0])
        self.valor_base = margen_ref - float(np.sum(self.shap_local(x_ref)[0]))

    def shap_local(self, X: np.ndarray) -> np.ndarray:
        """Matriz (n, variables) de valores SHAP en log-odds."""
        valores = self.tree_explainer.shap_values(X)
        if isinstance(valores, list):
            return valores[1]  # versiones viejas de shap devuelven una lista por clase
        return valores

    def importancia_global(self, df_muestra: pd.DataFrame = None, top: int = 15) -> pd.DataFrame:
        """Promedio de |SHAP| por variable (por defecto sobre test)."""
        if df_muestra is None:
            df_muestra = pd.read_csv(config.RUTA_TEST)

        X_muestra = self.preprocesador.transform(df_muestra)
        matriz_shap = self.shap_local(X_muestra)
        shap_medio_abs = np.mean(np.abs(matriz_shap), axis=0)

        df_imp = pd.DataFrame({
            "variable": self.nombres_variables,
            "shap_medio_abs": shap_medio_abs
        })

        df_imp["nombre"] = df_imp["variable"].apply(
            lambda x: NOMBRES_VARIABLES.get(x, x.replace("_", " ").title())
        )

        df_imp = df_imp.sort_values("shap_medio_abs", ascending=False).reset_index(drop=True)
        total_shap = df_imp["shap_medio_abs"].sum()
        df_imp["importancia_pct"] = (df_imp["shap_medio_abs"] / (total_shap if total_shap > 0 else 1.0)) * 100.0

        return df_imp.head(top)

    def evaluar_solicitud(self, solicitud: Dict[str, Any]) -> Dict[str, Any]:
        """PD, score, decisión, contribuciones SHAP y motivos de denegación para una solicitud."""
        df_solicitud = pd.DataFrame([solicitud])
        df_indices = agregar_indices(df_solicitud)
        X_t = self.preprocesador.transform(df_indices)

        probs = self.modelo.predict_proba(X_t)[0]
        pd_valor = float(probs[1])
        score = pd_a_score(pd_valor)
        decision, color, politica = decision_crediticia(pd_valor)

        monto = float(solicitud.get("monto_solicitado", 4500.0))
        perdida = perdida_esperada(pd_valor, monto, config.LGD)

        valores_shap = self.shap_local(X_t)[0]

        contribuciones = []
        for variable, valor in zip(self.nombres_variables, valores_shap):
            nombre = NOMBRES_VARIABLES.get(variable, variable.replace("_", " ").title())
            contribuciones.append({
                "variable": variable,
                "nombre": nombre,
                "shap": round(float(valor), 4),
                "shap_abs": abs(float(valor))
            })

        df_contrib = pd.DataFrame(contribuciones).sort_values("shap_abs", ascending=False)

        # SHAP negativo baja el riesgo, positivo lo sube
        factores_favorables = df_contrib[df_contrib["shap"] < 0].head(5).to_dict(orient="records")
        factores_riesgo = df_contrib[df_contrib["shap"] > 0].head(5).to_dict(orient="records")

        # Hasta 3 motivos, sin repetir texto
        motivos = []
        for factor in factores_riesgo:
            variable = factor["variable"]
            motivo = MOTIVOS_DENEGACION.get(
                variable,
                f"Impacto desfavorable en el indicador: {factor['nombre']} (Contribución: +{factor['shap']:.2f})"
            )
            if motivo not in motivos:
                motivos.append(motivo)
            if len(motivos) == 3:
                break

        if not motivos:
            motivos = ["No se registraron alertas severas de riesgo de impago en el perfil del solicitante."]

        dictamen = self._armar_dictamen(
            df_indices.iloc[0].to_dict(),
            pd_valor,
            score,
            decision,
            factores_favorables,
            factores_riesgo,
            motivos
        )

        return {
            "pd": round(pd_valor, 4),
            "pd_pct": round(pd_valor * 100.0, 2),
            "score": score,
            "decision": decision,
            "color": color,
            "politica": politica,
            "monto": monto,
            "perdida_esperada": perdida,
            "valor_base": round(self.valor_base, 4),
            "indices": {
                "Inclusión Digital": f"{df_indices['indice_inclusion_digital'].iloc[0]*100:.1f}%",
                "Resiliencia Financiera": f"{df_indices['indice_resiliencia_financiera'].iloc[0]*100:.1f}%",
                "Vulnerabilidad Liquidez": f"{df_indices['indice_vulnerabilidad_liquidez'].iloc[0]*100:.1f}%",
                "Cultura de Pago": f"{df_indices['indice_cultura_pago'].iloc[0]*100:.1f}%",
            },
            "contribuciones": df_contrib.head(10).to_dict(orient="records"),
            "factores_favorables": factores_favorables,
            "factores_riesgo": factores_riesgo,
            "motivos_denegacion": motivos,
            "dictamen": dictamen
        }

    def _armar_dictamen(
        self,
        datos: Dict[str, Any],
        pd_valor: float,
        score: int,
        decision: str,
        favorables: List[Dict[str, Any]],
        riesgos: List[Dict[str, Any]],
        motivos: List[str]
    ) -> Dict[str, Any]:
        """Resumen en texto para el analista."""
        puntos_fuertes = []
        for f in favorables[:3]:
            puntos_fuertes.append(f"{f['nombre']} (Atenúa riesgo en {abs(f['shap']):.2f} pts log-odds)")

        alertas = []
        for r in riesgos[:3]:
            alertas.append(f"{r['nombre']} (Aumenta riesgo en +{r['shap']:.2f} pts log-odds)")

        recomendaciones = []
        if decision == "APROBADO":
            recomendaciones.append("Aprobación recomendada. El perfil muestra solvencia transaccional y cultura de pago.")
            recomendaciones.append("Vincular pagos automáticos mediante billeteras móviles para conservar alta puntualidad.")
        elif decision == "REVISIÓN MANUAL":
            recomendaciones.append("Riesgo moderado. Requiere verificación de ingresos independientes y aval o garantía líquida.")
            recomendaciones.append("Comprobar extracto de billeteras digitales (Yape/Plin) de los últimos 3 meses.")
        else:
            recomendaciones.append("Motivos de denegación disponibles para el solicitante (Ley N.° 32294).")
            recomendaciones.append(f"Motivo prioritario: {motivos[0]}")
            recomendaciones.append("Recomendar regularización de cuentas de servicios y fomento del ahorro formal previo a una nueva postulación.")

        return {
            "resumen": f"Solicitante con Score Crediticio de {score}/1000 (PD estimada: {pd_valor*100:.2f}%), Dictamen: {decision}.",
            "puntos_fuertes": puntos_fuertes if puntos_fuertes else ["Perfil sin factores atenuantes destacados."],
            "alertas_riesgo": alertas if alertas else ["Sin factores de riesgo críticos identificados."],
            "recomendaciones": recomendaciones
        }

    def grafico_waterfall(self, solicitud: Dict[str, Any], max_variables: int = 10):
        df_solicitud = pd.DataFrame([solicitud])
        df_indices = agregar_indices(df_solicitud)
        X_t = self.preprocesador.transform(df_indices)

        valores_shap = self.shap_local(X_t)[0]

        # Mostrar los valores del cuestionario, no los escalados
        fila = df_indices.iloc[0]
        datos_mostrar = np.array([
            float(fila[f]) if f in self.preprocesador.numericas else float(v)
            for f, v in zip(self.nombres_variables, X_t[0])
        ])

        explicacion = shap.Explanation(
            values=valores_shap,
            base_values=self.valor_base,
            data=datos_mostrar,
            feature_names=[NOMBRES_VARIABLES.get(f, f) for f in self.nombres_variables]
        )

        fig = plt.figure(figsize=(9, 5.5), dpi=100)
        shap.plots.waterfall(explicacion, max_display=max_variables, show=False)
        plt.title(f"Explicabilidad Local TreeSHAP (Ley N.° 32294)", fontsize=11, fontweight="bold", pad=12)
        plt.tight_layout()
        return fig


def cargar_explicador() -> ExplicadorSHAP:
    preprocesador = Preprocesador.cargar(config.RUTA_PREPROCESADOR)
    modelo = ModeloXGBoost.cargar(config.RUTA_MODELO)
    return ExplicadorSHAP(modelo, preprocesador)


def generar_artefactos():
    """Beeswarm global y estadísticas SHAP sobre test."""
    explicador = cargar_explicador()
    df_test = pd.read_csv(config.RUTA_TEST)

    X_test = explicador.preprocesador.transform(df_test)
    matriz_shap = explicador.shap_local(X_test)

    nombres = [NOMBRES_VARIABLES.get(f, f.replace("_", " ").title()) for f in explicador.nombres_variables]

    plt.figure(figsize=(10, 6), dpi=120)
    shap.summary_plot(
        matriz_shap,
        X_test,
        feature_names=nombres,
        max_display=15,
        show=False
    )
    # shap pone las etiquetas en inglés
    fig = plt.gcf()
    fig.axes[0].set_xlabel("Valor SHAP (impacto en la log-odds de mora)")
    barra = fig.axes[-1]
    barra.set_ylabel("Valor de la variable")
    barra.set_yticklabels(["Bajo", "Alto"])
    plt.title("Ranking Global de Variables - TreeSHAP (Base Empírica SBS/CAF 2022)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    ruta_beeswarm = config.DIR_MODELOS / "shap_beeswarm.png"
    plt.savefig(ruta_beeswarm, bbox_inches="tight")
    plt.close()

    resumen = {
        "valor_base": float(explicador.valor_base),
        "importancia": explicador.importancia_global(df_test, top=20).to_dict(orient="records")
    }
    with open(config.RUTA_RESUMEN_SHAP, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    generar_artefactos()

    # prueba rápida con una solicitud
    ejemplo = {
        "edad": 34,
        "carga_familiar_adultos": 2,
        "carga_familiar_ninos": 1,
        "genero": "Mujer",
        "nivel_educacion": "Secundaria Completa",
        "departamento": "Lima / Callao",
        "ambito": "Urbano",
        "condicion_laboral": "Trabajador Independiente / Cuenta propia",
        "es_informal": "Informal / Sin RUC",
        "adopcion_yape_plin": "Sí",
        "tiene_billetera_activa": "Sí",
        "frecuencia_yape_plin": 3,
        "pago_celular_frecuencia": 4,
        "uso_whatsapp": 4,
        "uso_internet": 3,
        "preferencia_efectivo": 2,
        "confianza_herramientas_digitales": 4,
        "guarda_dinero_recibos": "Sí",
        "anota_recibos_vencer": "Sí",
        "puntualidad_pago_deudas": 4,
        "rango_ingreso": "De S/ 1,201 a S/ 2,400",
        "nivel_socioeconomico": "C",
        "ingresos_estables": "Sí",
        "planifica_presupuesto": "Sí",
        "anota_gastos": "Sí",
        "capacidad_afrontar_imprevisto": "Sí",
        "dificultad_emergencia_1000": 2,
        "meses_resiliencia_desempleo": 3,
        "sobra_dinero_fin_mes": 3,
        "ingreso_apenas_alcanza": 2,
        "percepcion_sobreendeudado": 2,
        "tiene_cuenta_ahorro": "Sí",
        "tiene_tarjeta_credito": "No",
        "tiene_prestamo_personal": "No",
        "ha_usado_prestamista": "No",
        "ha_usado_pandero_junta": "No",
        "credito_formal_negado": "No",
        "monto_solicitado": 4000.0
    }

    res = cargar_explicador().evaluar_solicitud(ejemplo)
    print(f"{res['decision']} | score {res['score']} | PD {res['pd_pct']}%")
    for r in res["motivos_denegacion"]:
        print(" -", r)
