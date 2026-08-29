"""
==============================================================================
PROYECTO DE TESIS - UNIVERSIDAD DE LIMA
Carrera: Ingeniería de Sistemas
Título: Modelo de evaluación de riesgo crediticio para microfinanzas e
        inclusión financiera mediante XGBoost e Inteligencia Artificial
        Explicable (SHAP)
Archivo: src/explainability.py
Descripción: Módulo de Inteligencia Artificial Explicable (XAI) basado en
             valores de Shapley (TreeSHAP) para explicabilidad global (portafolio)
             y local (solicitud individual), con generación de reportes narrativos
             para comités de riesgo y análisis de sensibilidad What-If.
==============================================================================
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path
import sys

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


# Diccionario amigable de nombres de características para el analista de crédito
FEATURE_FRIENDLY_NAMES = {
    "ratio_cuota_ingreso": "Ratio Cuota/Ingreso (DTI)",
    "ratio_cobertura_deuda": "Cobertura de Deuda (DSCR)",
    "flujo_caja_libre_mensual": "Flujo de Caja Disponible (S/)",
    "ingreso_mensual_neto": "Ingreso Mensual Neto (S/)",
    "gasto_mensual_operativo": "Gasto Operativo Mensual (S/)",
    "monto_solicitado": "Monto Solicitado (S/)",
    "plazo_meses": "Plazo Solicitado (Meses)",
    "tasa_interes_anual": "Tasa de Interés TEA (%)",
    "antiguedad_negocio_meses": "Antigüedad del Negocio (Meses)",
    "variabilidad_ingresos_cv": "Variabilidad de Ventas / Estacionalidad",
    "num_consultas_centrales_ult_6m": "Consultas SBS/Infocorp (Últ. 6 meses)",
    "puntualidad_servicios_basicos": "Puntualidad Luz/Agua (12 meses)",
    "volumen_transacciones_digitales": "Volumen Yape/Plin Mensual (S/)",
    "frecuencia_transacciones_yape_mes": "Frecuencia Transaccional Digital",
    "antiguedad_celular_meses": "Estabilidad Línea Celular (Meses)",
    "indice_estabilidad_negocio": "Índice Estabilidad del Negocio",
    "score_comportamiento_alternativo": "Score de Inclusión Financiera",
    "edad": "Edad del Solicitante",
    "es_informal_1": "Negocio Informal / Sin RUC",
    "es_informal_0": "Negocio Formalizado",
    "posee_garantia_o_aval_1": "Posee Garantía o Aval",
    "posee_garantia_o_aval_0": "Sin Garantía / Aval",
}


class CreditRiskExplainer:
    """
    Motor de explicabilidad algorítmica para el modelo de riesgo crediticio.
    Calcula contribuciones marginales de Shapley (SHAP) a nivel global y local.
    """

    def __init__(
        self,
        model_wrapper: CreditRiskXGBoostWrapper,
        preprocessor: CreditDataPreprocessor,
        background_data: np.ndarray = None
    ):
        self.model_wrapper = model_wrapper
        self.preprocessor = preprocessor
        self.feature_names = preprocessor.get_feature_names_out()
        self.shap_explainer = None
        self.shap_available = False

        # Intentar inicializar TreeSHAP de la librería oficial shap
        try:
            import shap
            if hasattr(model_wrapper, "native_xgboost_available") and model_wrapper.native_xgboost_available:
                self.shap_explainer = shap.TreeExplainer(model_wrapper.model)
                self.shap_available = True
        except ImportError:
            self.shap_available = False

        # Base value (esperanza del modelo)
        self.expected_value = float(model_wrapper.bias) if hasattr(model_wrapper, "bias") else 0.0

    def compute_local_shap(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Calcula los valores SHAP locales para una o más observaciones.
        Retorna un array (n_samples, n_features).
        """
        if self.shap_available and self.shap_explainer is not None:
            raw_shap = self.shap_explainer.shap_values(X_transformed)
            if isinstance(raw_shap, list):
                return raw_shap[1]  # Clase positiva (default)
            return raw_shap
        else:
            # Descomposición aditiva exacta para el modelo entrenado
            weights = self.model_wrapper.weights if self.model_wrapper.weights is not None else np.zeros(X_transformed.shape[1])
            shap_values = X_transformed * weights.reshape(1, -1)
            return shap_values

    def get_global_feature_importance(self, df_test: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        """
        Calcula la importancia global de características promediando |SHAP| sobre el dataset de test.
        """
        X_test = self.preprocessor.transform(df_test)
        shap_matrix = self.compute_local_shap(X_test)
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

        df_imp = pd.DataFrame({
            "feature_raw": self.feature_names,
            "mean_abs_shap": mean_abs_shap
        })

        # Asignar nombres amigables legibles
        df_imp["feature_name"] = df_imp["feature_raw"].apply(
            lambda x: FEATURE_FRIENDLY_NAMES.get(x, x.replace("_", " ").title())
        )

        # Agrupar si es necesario y ordenar
        df_imp = df_imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        total_shap = df_imp["mean_abs_shap"].sum()
        df_imp["importance_pct"] = (df_imp["mean_abs_shap"] / (total_shap if total_shap > 0 else 1)) * 100.0

        return df_imp.head(top_n)

    def explain_single_applicant(self, applicant_raw_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera la evaluación crediticia completa y la explicabilidad local
        para una solicitud de microcrédito individual.

        Args:
            applicant_raw_dict: Diccionario con los datos del solicitante.

        Returns:
            Dict con probabilidad PD, score (0-1000), decisión, pérdida esperada,
            valores SHAP locales y narrativa explicativa para el analista.
        """
        df_single = pd.DataFrame([applicant_raw_dict])
        df_enriched = add_engineered_features(df_single)
        X_trans = self.preprocessor.transform(df_enriched)

        # 1. Inferencia Predictiva
        prob_array = self.model_wrapper.predict_proba(X_trans)[0]
        pd_value = float(prob_array[1])
        score = convert_pd_to_credit_score(pd_value)
        decision, badge_color, policy_desc = get_credit_decision(pd_value)
        
        monto = float(applicant_raw_dict.get("monto_solicitado", 5000.0))
        expected_loss = calculate_expected_loss(pd_value, monto, config.DEFAULT_LGD)

        # 2. Cálculo de Valores SHAP Locales
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

        # Ordenar por magnitud de impacto
        df_contribs = pd.DataFrame(feature_contributions).sort_values("abs_shap", ascending=False)
        
        # Factores protectores (reducen riesgo, SHAP negativo en log-odds)
        factores_favorables = df_contribs[df_contribs["shap_value"] < 0].head(5).to_dict(orient="records")
        # Factores de riesgo (incrementan riesgo, SHAP positivo)
        factores_riesgo = df_contribs[df_contribs["shap_value"] > 0].head(5).to_dict(orient="records")

        # 3. Generación de Narrativa Explicativa en Lenguaje de Negocio
        narrativa = self._generate_business_narrative(
            df_enriched.iloc[0].to_dict(),
            pd_value,
            score,
            decision,
            factores_favorables,
            factores_riesgo
        )

        return {
            "pd_value": round(pd_value, 4),
            "pd_pct": round(pd_value * 100.0, 2),
            "score_crediticio": score,
            "decision": decision,
            "badge_color": badge_color,
            "politica_descripcion": policy_desc,
            "monto_solicitado": monto,
            "perdida_esperada": expected_loss,
            "ratios_clave": {
                "DTI (Cuota/Ingreso)": f"{df_enriched['ratio_cuota_ingreso'].iloc[0]*100:.1f}%",
                "DSCR (Cobertura)": f"{df_enriched['ratio_cobertura_deuda'].iloc[0]:.2f}x",
                "Cuota Estimada": f"S/ {df_enriched['cuota_mensual'].iloc[0]:,.2f}",
                "Flujo Disponible": f"S/ {df_enriched['flujo_caja_libre_mensual'].iloc[0]:,.2f}",
                "Score Inclusión Alternativo": f"{df_enriched['score_comportamiento_alternativo'].iloc[0]:.2f}/1.00"
            },
            "shap_features": df_contribs.head(10).to_dict(orient="records"),
            "factores_favorables": factores_favorables,
            "factores_riesgo": factores_riesgo,
            "narrativa_comite": narrativa
        }

    def _generate_business_narrative(
        self,
        applicant_data: Dict[str, Any],
        pd_val: float,
        score: int,
        decision: str,
        favorables: List[Dict[str, Any]],
        riesgos: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Genera el informe ejecutivo en lenguaje bancario para el comité de créditos.
        """
        dti = applicant_data.get("ratio_cuota_ingreso", 0.0) * 100.0
        dscr = applicant_data.get("ratio_cobertura_deuda", 0.0)
        monto = applicant_data.get("monto_solicitado", 0.0)
        plazo = applicant_data.get("plazo_meses", 12)
        antig = applicant_data.get("antiguedad_negocio_meses", 0)
        puntualidad = applicant_data.get("puntualidad_servicios_basicos", 0.0) * 100.0

        puntos_fuertes = []
        if puntualidad >= 80.0:
            puntos_fuertes.append(f"Excelente comportamiento de pago en servicios públicos ({puntualidad:.0f}% de puntualidad en los últimos 12 meses).")
        if antig >= 24:
            puntos_fuertes.append(f"Estabilidad comercial consolidada ({antig} meses de operación continua del negocio).")
        if dscr >= 1.5:
            puntos_fuertes.append(f"Solidez en cobertura de deuda (Flujo libre cubre {dscr:.2f} veces la cuota estimada).")
        if applicant_data.get("volumen_transacciones_digitales", 0) > 1000:
            puntos_fuertes.append(f"Actividad transaccional recurrente en billeteras digitales (Yape/Plin: S/ {applicant_data['volumen_transacciones_digitales']:,.0f}/mes).")

        alertas_riesgo = []
        if dti > 35.0:
            alertas_riesgo.append(f"Ratio de endeudamiento DTI elevado ({dti:.1f}% de sus ingresos netos destinados a la cuota).")
        if dscr < 1.15:
            alertas_riesgo.append(f"Ajustado margen de liquidez (DSCR de {dscr:.2f}x).")
        if applicant_data.get("num_consultas_centrales_ult_6m", 0) >= 3:
            alertas_riesgo.append(f"Múltiples consultas recientes en centrales de riesgo ({applicant_data['num_consultas_centrales_ult_6m']} consultas en 6 meses).")
        if applicant_data.get("variabilidad_ingresos_cv", 0) > 0.40:
            alertas_riesgo.append(f"Alta estacionalidad y variabilidad en los flujos de ventas del negocio.")

        # Recomendaciones operativas para el analista
        recomendaciones = []
        if decision == "APROBADO":
            recomendaciones.append("Desembolso recomendado bajo las condiciones solicitadas.")
            recomendaciones.append("Sugerir canal de pago digital automático para fidelización.")
        elif decision == "REVISIÓN MANUAL":
            recomendaciones.append(f"Proponer ampliación del plazo de {plazo} a {min(plazo + 6, 48)} meses para reducir la cuota mensual.")
            recomendaciones.append("Solicitar aval solidario o verificación presencial del inventario del negocio.")
        else:
            monto_sugerido = max(1000, round(monto * 0.65, -2))
            recomendaciones.append(f"Reestructurar solicitud: Reducir el monto solicitado de S/ {monto:,.0f} a S/ {monto_sugerido:,.0f}.")
            recomendaciones.append("Recomendar regularización de consultas en centrales y estabilización de ventas en los próximos 3 meses.")

        return {
            "resumen_ejecutivo": f"El solicitante presenta un Score Crediticio de {score}/1000 con una Probabilidad de Impago estimada de {pd_val*100:.2f}%, ubicándose en categoría de decisión: {decision}.",
            "puntos_fuertes": puntos_fuertes if puntos_fuertes else ["No se identificaron fortalezas significativas."],
            "alertas_riesgo": alertas_riesgo if alertas_riesgo else ["Perfil dentro de los parámetros estándar de riesgo."],
            "recomendaciones_analista": recomendaciones
        }

    def simulate_what_if(
        self,
        original_dict: Dict[str, Any],
        modified_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compara el impacto de modificar variables de la solicitud (Análisis de Sensibilidad What-If).
        """
        eval_original = self.explain_single_applicant(original_dict)
        eval_modified = self.explain_single_applicant(modified_dict)

        delta_pd = eval_modified["pd_pct"] - eval_original["pd_pct"]
        delta_score = eval_modified["score_crediticio"] - eval_original["score_crediticio"]
        delta_el = eval_modified["perdida_esperada"] - eval_original["perdida_esperada"]

        return {
            "original": eval_original,
            "modificado": eval_modified,
            "delta_pd_pct": round(delta_pd, 2),
            "delta_score": delta_score,
            "delta_perdida_esperada": round(delta_el, 2),
            "cambio_decision": (eval_original["decision"] != eval_modified["decision"]),
            "decision_anterior": eval_original["decision"],
            "decision_nueva": eval_modified["decision"]
        }


def get_default_explainer() -> CreditRiskExplainer:
    """Instancia y retorna el explicador configurado con los modelos persistidos."""
    preprocessor = CreditDataPreprocessor.load(config.PREPROCESSOR_PATH)
    model = CreditRiskXGBoostWrapper.load(config.XGBOOST_MODEL_JSON)
    return CreditRiskExplainer(model, preprocessor)


if __name__ == "__main__":
    # Prueba rápida del explicador
    explainer = get_default_explainer()
    sample_applicant = {
        "edad": 38,
        "genero": "F",
        "estado_civil": "Casado(a)",
        "nivel_educacion": "Secundaria Completa",
        "tipo_vivienda": "Propia",
        "region_peru": "Lima/Callao",
        "rubro_negocio": "Comercio Minorista / Bodega",
        "antiguedad_negocio_meses": 36,
        "es_informal": 1,
        "ingreso_mensual_neto": 3800.0,
        "gasto_mensual_operativo": 2100.0,
        "monto_solicitado": 7500.0,
        "plazo_meses": 18,
        "tasa_interes_anual": 0.38,
        "num_consultas_centrales_ult_6m": 1,
        "posee_garantia_o_aval": 0,
        "puntualidad_servicios_basicos": 0.92,
        "volumen_transacciones_digitales": 1450.0,
        "frecuencia_transacciones_yape_mes": 42,
        "antiguedad_celular_meses": 28,
        "variabilidad_ingresos_cv": 0.18
    }

    result = explainer.explain_single_applicant(sample_applicant)
    print("\n--- [Prueba de Evaluación Crediticia e Interpretabilidad Local] ---")
    print(f"Decisión: {result['decision']} (Score: {result['score_crediticio']}/1000 | PD: {result['pd_pct']}%)")
    print(f"Pérdida Esperada: S/ {result['perdida_esperada']:,.2f}")
    print("\nNarrativa del Comité:")
    print(" ", result["narrativa_comite"]["resumen_ejecutivo"])
    print(" Puntos Fuertes:", result["narrativa_comite"]["puntos_fuertes"])
    print(" Alertas de Riesgo:", result["narrativa_comite"]["alertas_riesgo"])
    print(" Recomendaciones:", result["narrativa_comite"]["recomendaciones_analista"])
