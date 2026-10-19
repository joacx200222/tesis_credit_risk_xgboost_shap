"""
Descripción: Módulo de Inteligencia Artificial Explicable (XAI) basado en
             valores de Shapley (TreeSHAP) para explicabilidad global (portafolio)
             y local (solicitud individual), con extracción de razones de
             acción adversa (Adverse Action Reasons) conforme a la Res. SBS 3780-2021.
"""

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import shap

# Importar configuración, preprocesamiento y evaluación
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.data_preprocessing import CreditDataPreprocessor
from src.model_training import CreditRiskXGBoostWrapper
from src.feature_engineering import add_engineered_features
from src.model_evaluation import (
    calculate_expected_loss,
    convert_pd_to_credit_score,
    get_credit_decision
)


# Diccionario de nombres amigables para el analista de riesgo y comités de crédito
FEATURE_FRIENDLY_NAMES = {
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


# Mapeo oficial de factores SHAP a Razones de Acción Adversa (SBS Res. N.° 3780-2021)
ADVERSE_ACTION_REASON_MAP = {
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


class CreditRiskExplainer:
    """
    Motor de explicabilidad algorítmica para el modelo de riesgo crediticio SBS.
    Implementa TreeSHAP con soporte para beeswarm global, waterfall local y
    dictamen auditable conforme a la Resolución SBS N.° 3780-2021.
    """

    def __init__(
        self,
        model_wrapper: CreditRiskXGBoostWrapper,
        preprocessor: CreditDataPreprocessor
    ):
        self.model_wrapper = model_wrapper
        self.preprocessor = preprocessor
        self.feature_names = preprocessor.get_feature_names_out()

        # Instanciar TreeSHAP sobre el estimador nativo de XGBoost
        native_model = model_wrapper.model
        self.tree_explainer = shap.TreeExplainer(native_model)

        # Valor base E[f(x)] derivado del axioma de eficiencia: f(x) = E[f(x)] + sum(phi_i).
        # Algunas combinaciones de versiones shap/xgboost reportan expected_value = 0
        # al no leer correctamente el base_score del booster, por lo que se obtiene
        # directamente del margen del modelo para garantizar la aditividad exacta.
        x_ref = np.zeros((1, len(self.feature_names)))
        margin_ref = float(native_model.predict(x_ref, output_margin=True)[0])
        self.expected_value = margin_ref - float(np.sum(self.compute_local_shap(x_ref)[0]))

    def compute_local_shap(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Calcula los valores SHAP marginales phi_i para una o más observaciones.
        Retorna un ndarray (n_samples, n_features) en escala log-odds de default.
        """
        raw_shap = self.tree_explainer.shap_values(X_transformed)
        if isinstance(raw_shap, list):
            return raw_shap[1]  # Si fuera multiclase, clase 1
        return raw_shap

    def get_global_feature_importance(self, df_sample: pd.DataFrame = None, top_n: int = 15) -> pd.DataFrame:
        """
        Calcula la importancia global de características promediando |SHAP| sobre una muestra.
        """
        if df_sample is None:
            df_sample = pd.read_csv(config.PROCESSED_TEST_PATH)

        X_sample = self.preprocessor.transform(df_sample)
        shap_matrix = self.compute_local_shap(X_sample)
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

        df_imp = pd.DataFrame({
            "feature_raw": self.feature_names,
            "mean_abs_shap": mean_abs_shap
        })

        df_imp["feature_name"] = df_imp["feature_raw"].apply(
            lambda x: FEATURE_FRIENDLY_NAMES.get(x, x.replace("_", " ").title())
        )

        df_imp = df_imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        total_shap = df_imp["mean_abs_shap"].sum()
        df_imp["importance_pct"] = (df_imp["mean_abs_shap"] / (total_shap if total_shap > 0 else 1.0)) * 100.0

        return df_imp.head(top_n)

    def explain_single_applicant(self, applicant_raw_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera la evaluación crediticia institucional, cálculo de score (0-1000),
        probabilidad de default PD, valores TreeSHAP phi_i y razones de acción adversa SBS.
        """
        df_single = pd.DataFrame([applicant_raw_dict])
        df_enriched = add_engineered_features(df_single)
        X_trans = self.preprocessor.transform(df_enriched)

        # 1. Inferencia Predictiva
        prob_array = self.model_wrapper.predict_proba(X_trans)[0]
        pd_value = float(prob_array[1])
        score = convert_pd_to_credit_score(pd_value)
        decision, badge_color, policy_desc = get_credit_decision(pd_value)
        
        monto_estimado = float(applicant_raw_dict.get("monto_solicitado", 4500.0))
        expected_loss = calculate_expected_loss(pd_value, monto_estimado, config.DEFAULT_LGD)

        # 2. TreeSHAP Local
        shap_vals = self.compute_local_shap(X_trans)[0]

        feature_contributions = []
        for feat_raw, s_val in zip(self.feature_names, shap_vals):
            feat_name = FEATURE_FRIENDLY_NAMES.get(feat_raw, feat_raw.replace("_", " ").title())
            feature_contributions.append({
                "feature_raw": feat_raw,
                "feature_name": feat_name,
                "shap_value": round(float(s_val), 4),
                "abs_shap": abs(float(s_val))
            })

        df_contribs = pd.DataFrame(feature_contributions).sort_values("abs_shap", ascending=False)
        
        # Factores protectores (reducen riesgo: SHAP negativo)
        factores_favorables = df_contribs[df_contribs["shap_value"] < 0].head(5).to_dict(orient="records")
        # Factores de riesgo (aumentan riesgo: SHAP positivo)
        factores_riesgo = df_contribs[df_contribs["shap_value"] > 0].head(5).to_dict(orient="records")

        # 3. Extracción de las 3 Razones Principales de Acción Adversa (SBS Res. 3780-2021)
        adverse_reasons = []
        for item in factores_riesgo:
            feat = item["feature_raw"]
            reason_text = ADVERSE_ACTION_REASON_MAP.get(
                feat,
                f"Impacto desfavorable en el indicador: {item['feature_name']} (Contribución: +{item['shap_value']:.2f})"
            )
            if reason_text not in adverse_reasons:
                adverse_reasons.append(reason_text)
            if len(adverse_reasons) == 3:
                break

        if not adverse_reasons:
            adverse_reasons = ["No se registraron alertas severas de riesgo de impago en el perfil del solicitante."]

        # 4. Narrativa de Comité
        narrativa = self._generate_business_narrative(
            df_enriched.iloc[0].to_dict(),
            pd_value,
            score,
            decision,
            factores_favorables,
            factores_riesgo,
            adverse_reasons
        )

        return {
            "pd_value": round(pd_value, 4),
            "pd_pct": round(pd_value * 100.0, 2),
            "score_crediticio": score,
            "decision": decision,
            "badge_color": badge_color,
            "politica_descripcion": policy_desc,
            "monto_referencial": monto_estimado,
            "perdida_esperada": expected_loss,
            "expected_value": round(self.expected_value, 4),
            "ratios_clave": {
                "Inclusión Digital": f"{df_enriched['indice_inclusion_digital'].iloc[0]*100:.1f}%",
                "Resiliencia Financiera": f"{df_enriched['indice_resiliencia_financiera'].iloc[0]*100:.1f}%",
                "Vulnerabilidad Liquidez": f"{df_enriched['indice_vulnerabilidad_liquidez'].iloc[0]*100:.1f}%",
                "Cultura de Pago": f"{df_enriched['indice_cultura_pago'].iloc[0]*100:.1f}%",
            },
            "shap_features": df_contribs.head(10).to_dict(orient="records"),
            "factores_favorables": factores_favorables,
            "factores_riesgo": factores_riesgo,
            "razones_accion_adversa_sbs": adverse_reasons,
            "narrativa_comite": narrativa
        }

    def _generate_business_narrative(
        self,
        applicant_data: Dict[str, Any],
        pd_val: float,
        score: int,
        decision: str,
        favorables: List[Dict[str, Any]],
        riesgos: List[Dict[str, Any]],
        adverse_reasons: List[str]
    ) -> Dict[str, Any]:
        """
        Construye el dictamen auditable individual para el comité de riesgo.
        """
        puntos_fuertes = []
        for f in favorables[:3]:
            puntos_fuertes.append(f"{f['feature_name']} (Atenúa riesgo en {abs(f['shap_value']):.2f} pts log-odds)")

        alertas_riesgo = []
        for r in riesgos[:3]:
            alertas_riesgo.append(f"{r['feature_name']} (Aumenta riesgo en +{r['shap_value']:.2f} pts log-odds)")

        recomendaciones = []
        if decision == "APROBADO":
            recomendaciones.append("Aprobación recomendada. El perfil muestra solvencia transaccional y cultura de pago.")
            recomendaciones.append("Vincular pagos automáticos mediante billeteras móviles para conservar alta puntualidad.")
        elif decision == "REVISIÓN MANUAL":
            recomendaciones.append("Riesgo moderado. Requiere verificación de ingresos independientes y aval o garantía líquida.")
            recomendaciones.append("Comprobar extracto de billeteras digitales (Yape/Plin) de los últimos 3 meses.")
        else:
            recomendaciones.append("Denegación fundamentada bajo Resolución SBS N.° 3780-2021.")
            recomendaciones.append(f"Motivo prioritario: {adverse_reasons[0]}")
            recomendaciones.append("Recomendar regularización de cuentas de servicios y fomento del ahorro formal previo a una nueva postulación.")

        return {
            "resumen_ejecutivo": f"Solicitante con Score Crediticio de {score}/1000 (PD estimada: {pd_val*100:.2f}%), Dictamen: {decision}.",
            "puntos_fuertes": puntos_fuertes if puntos_fuertes else ["Perfil sin factores atenuantes destacados."],
            "alertas_riesgo": alertas_riesgo if alertas_riesgo else ["Sin factores de riesgo críticos identificados."],
            "recomendaciones_analista": recomendaciones
        }

    def generate_waterfall_plot_figure(self, applicant_raw_dict: Dict[str, Any], max_display: int = 10):
        """
        Genera una figura Matplotlib del Waterfall plot de TreeSHAP para la solicitud individual.
        Optimizado para renderizarse en menos de 0.2 segundos en Streamlit.
        """
        df_single = pd.DataFrame([applicant_raw_dict])
        df_enriched = add_engineered_features(df_single)
        X_trans = self.preprocessor.transform(df_enriched)

        shap_vals = self.compute_local_shap(X_trans)[0]

        # Mostrar valores originales del cuestionario (no escalados) junto a cada variable
        raw_row = df_enriched.iloc[0]
        display_data = np.array([
            float(raw_row[f]) if f in self.preprocessor.numerical_features else float(v)
            for f, v in zip(self.feature_names, X_trans[0])
        ])

        # Crear Explanation object de shap
        explanation = shap.Explanation(
            values=shap_vals,
            base_values=self.expected_value,
            data=display_data,
            feature_names=[FEATURE_FRIENDLY_NAMES.get(f, f) for f in self.feature_names]
        )

        fig = plt.figure(figsize=(9, 5.5), dpi=100)
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.title(f"Explicabilidad Local TreeSHAP (Resolución SBS N.° 3780-2021)", fontsize=11, fontweight="bold", pad=12)
        plt.tight_layout()
        return fig


def get_default_explainer() -> CreditRiskExplainer:
    """Carga y retorna el explicador configurado."""
    preprocessor = CreditDataPreprocessor.load(config.PREPROCESSOR_PATH)
    model = CreditRiskXGBoostWrapper.load(config.XGBOOST_MODEL_JSON)
    return CreditRiskExplainer(model, preprocessor)


def generate_and_save_artifacts():
    """Genera y guarda el beeswarm plot global y las estadísticas de explicabilidad."""
    print("--- [Generando Artefactos de Explicabilidad TreeSHAP SBS] ---")
    explainer = get_default_explainer()
    df_test = pd.read_csv(config.PROCESSED_TEST_PATH)

    X_test = explainer.preprocessor.transform(df_test)
    shap_matrix = explainer.compute_local_shap(X_test)

    friendly_names = [FEATURE_FRIENDLY_NAMES.get(f, f) for f in explainer.feature_names]

    # 1. Beeswarm Summary Plot
    plt.figure(figsize=(10, 6), dpi=120)
    shap.summary_plot(
        shap_matrix,
        X_test,
        feature_names=friendly_names,
        max_display=15,
        show=False
    )
    plt.title("Ranking Global de Variables - TreeSHAP (Base Empírica SBS/CAF 2022)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    beeswarm_path = config.MODELS_DIR / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Beeswarm plot guardado en: {beeswarm_path}")

    # 2. Guardar estadísticas de resumen SHAP
    stats = {
        "expected_value": explainer.expected_value,
        "feature_importance": explainer.get_global_feature_importance(df_test, top_n=20).to_dict(orient="records")
    }
    joblib.dump(stats, config.SHAP_SUMMARY_PATH)
    print(f"[OK] Estadísticas SHAP guardadas en: {config.SHAP_SUMMARY_PATH}")


if __name__ == "__main__":
    generate_and_save_artifacts()
    
    # Test de solicitud individual
    sample = {
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
    
    exp = get_default_explainer()
    res = exp.explain_single_applicant(sample)
    print("\n--- Test Individual Explicabilidad ---")
    print(f"Decisión: {res['decision']} | Score: {res['score_crediticio']} | PD: {res['pd_pct']}%")
    print("Razones Acción Adversa:", res["razones_accion_adversa_sbs"])
