"""
Descripción: Pruebas unitarias y de integración del pipeline completo de scoring
             con la base empírica oficial SBS/CAF 2022 y TreeSHAP.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Añadir directorio raíz al path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.feature_engineering import load_and_extract_sbs_dataset, add_engineered_features
from src.data_preprocessing import CreditDataPreprocessor, stratified_split_dataset
from src.model_training import CreditRiskXGBoostWrapper
from src.model_evaluation import (
    compute_classification_metrics,
    calculate_expected_loss,
    convert_pd_to_credit_score,
    get_credit_decision,
    compute_kolmogorov_smirnov
)
from src.explainability import CreditRiskExplainer, get_default_explainer


class TestCreditRiskPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Carga y procesa un subconjunto de prueba de la base empírica SBS/CAF 2022."""
        cls.df_sbs = load_and_extract_sbs_dataset(config.RAW_DTA_PATH)

    def test_sbs_dataset_structure(self):
        """Verifica que el dataset SBS cargue las dimensiones y columnas correctas."""
        self.assertEqual(len(self.df_sbs), 1201)
        self.assertIn(config.TARGET_COLUMN, self.df_sbs.columns)
        self.assertIn("adopcion_yape_plin", self.df_sbs.columns)
        self.assertIn("frecuencia_yape_plin", self.df_sbs.columns)
        self.assertIn("puntualidad_pago_deudas", self.df_sbs.columns)
        self.assertIn("guarda_dinero_recibos", self.df_sbs.columns)
        
        # Tasa de morosidad empírica
        default_rate = self.df_sbs[config.TARGET_COLUMN].mean()
        self.assertAlmostEqual(default_rate, 0.2506, delta=0.01)

    def test_feature_engineering_indices(self):
        """Verifica el cálculo de los 4 índices sintéticos derivados."""
        for col in config.NUMERICAL_DERIVED_FEATURES:
            self.assertIn(col, self.df_sbs.columns)
            # Todos los índices deben estar normalizados entre 0.0 y 1.0
            self.assertTrue((self.df_sbs[col] >= 0.0).all())
            self.assertTrue((self.df_sbs[col] <= 1.0).all())

    def test_preprocessor_fit_transform(self):
        """Verifica el ajuste y transformación del preprocesador con codificación One-Hot."""
        preprocessor = CreditDataPreprocessor()
        X_trans = preprocessor.fit_transform(self.df_sbs)
        
        self.assertEqual(X_trans.shape[0], 1201)
        self.assertGreater(X_trans.shape[1], 50)
        self.assertFalse(np.isnan(X_trans).any())
        self.assertEqual(len(preprocessor.get_feature_names_out()), X_trans.shape[1])

    def test_model_training_and_inference(self):
        """Verifica el entrenamiento y predicción probabilística de XGBoost."""
        preprocessor = CreditDataPreprocessor()
        X_trans = preprocessor.fit_transform(self.df_sbs)
        y = self.df_sbs[config.TARGET_COLUMN].values

        model = CreditRiskXGBoostWrapper()
        model.fit(X_trans, y, feature_names=preprocessor.get_feature_names_out())
        
        probs = model.predict_proba(X_trans)
        self.assertEqual(probs.shape, (1201, 2))
        self.assertTrue((probs[:, 1] >= 0.0).all() and (probs[:, 1] <= 1.0).all())

    def test_metrics_and_financial_calculations(self):
        """Verifica métricas de clasificación, KS y pérdida esperada."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.75])
        
        metrics = compute_classification_metrics(y_true, y_prob)
        self.assertGreaterEqual(metrics["auc_roc"], 0.90)
        
        # Pérdida Esperada EL = PD * EAD * LGD
        el = calculate_expected_loss(0.20, 10000.0, 0.50)
        self.assertEqual(el, 1000.0)

        # Conversión a Score (0 - 1000)
        score = convert_pd_to_credit_score(0.20)
        self.assertEqual(score, 800)

        # Decisión crediticia
        decision, badge, desc = get_credit_decision(0.15)
        self.assertEqual(decision, "APROBADO")

        decision_med, _, _ = get_credit_decision(0.35)
        self.assertEqual(decision_med, "REVISIÓN MANUAL")

        decision_high, _, _ = get_credit_decision(0.60)
        self.assertEqual(decision_high, "RECHAZADO")

    def test_treeshap_explainer_local_and_adverse_actions(self):
        """Verifica la generación de explicaciones TreeSHAP y razones de acción adversa SBS."""
        explainer = get_default_explainer()
        sample_dict = self.df_sbs.iloc[0].to_dict()
        
        result = explainer.explain_single_applicant(sample_dict)
        self.assertIn("score_crediticio", result)
        self.assertIn("pd_value", result)
        self.assertIn("narrativa_comite", result)
        self.assertIn("shap_features", result)
        self.assertIn("razones_accion_adversa_sbs", result)
        self.assertTrue(len(result["razones_accion_adversa_sbs"]) >= 1)

    def test_treeshap_additivity(self):
        """Verifica el axioma de eficiencia: E[f(x)] + sum(phi_i) = f(x) en log-odds."""
        explainer = get_default_explainer()
        df_test = pd.read_csv(config.PROCESSED_TEST_PATH)
        X_test = explainer.preprocessor.transform(df_test)

        shap_matrix = explainer.compute_local_shap(X_test)
        margin = explainer.model_wrapper.model.predict(X_test, output_margin=True)
        error = np.abs(explainer.expected_value + shap_matrix.sum(axis=1) - margin).max()
        self.assertLess(error, 1e-4)

    def test_credit_policy_thresholds(self):
        """Verifica los bordes de la política: PD <= 0.30 aprueba, PD <= 0.45 revisa, PD > 0.45 rechaza."""
        self.assertEqual(get_credit_decision(0.30)[0], "APROBADO")
        self.assertEqual(get_credit_decision(0.45)[0], "REVISIÓN MANUAL")
        self.assertEqual(get_credit_decision(0.4501)[0], "RECHAZADO")

    def test_sample_profiles_match_their_label(self):
        """Verifica que cada perfil de demostración del prototipo obtenga la decisión que anuncia."""
        from src.app import create_sample_profiles
        explainer = get_default_explainer()
        esperado = {"Aprobado": "APROBADO", "Revisión Manual": "REVISIÓN MANUAL", "Rechazado": "RECHAZADO"}
        for nombre, perfil in create_sample_profiles().items():
            for etiqueta, decision in esperado.items():
                if etiqueta in nombre:
                    result = explainer.explain_single_applicant(perfil)
                    self.assertEqual(result["decision"], decision, msg=nombre)


if __name__ == "__main__":
    unittest.main()
