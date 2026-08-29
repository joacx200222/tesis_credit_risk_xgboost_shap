"""
==============================================================================
PROYECTO DE TESIS - UNIVERSIDAD DE LIMA
Carrera: Ingeniería de Sistemas
Archivo: tests/test_pipeline.py
Descripción: Pruebas unitarias y de integración del pipeline completo de scoring.
==============================================================================
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Añadir directorio raíz al path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.generate_synthetic_data import generate_microfinance_dataset
from src.feature_engineering import add_engineered_features, calculate_monthly_installment
from src.data_preprocessing import CreditDataPreprocessor, stratified_split_dataset
from src.model_training import CreditRiskXGBoostWrapper, StandaloneLogisticModel
from src.model_evaluation import (
    compute_classification_metrics,
    calculate_expected_loss,
    convert_pd_to_credit_score,
    get_credit_decision
)
from src.explainability import CreditRiskExplainer


class TestCreditRiskPipeline(unittest.TestCase):

    def setUp(self):
        """Configuración de datos de prueba para tests."""
        self.df_sample = generate_microfinance_dataset(n_samples=200, random_state=42)
        self.df_enriched = add_engineered_features(self.df_sample)

    def test_synthetic_data_generation(self):
        """Verifica que el generador de datos cree las columnas requeridas y tipos correctos."""
        self.assertEqual(len(self.df_sample), 200)
        self.assertIn(config.TARGET_COLUMN, self.df_sample.columns)
        self.assertIn("edad", self.df_sample.columns)
        self.assertIn("volumen_transacciones_digitales", self.df_sample.columns)
        
        # Verificar tasa de mora razonable
        default_rate = self.df_sample[config.TARGET_COLUMN].mean()
        self.assertTrue(0.05 <= default_rate <= 0.25)

    def test_feature_engineering_ratios(self):
        """Verifica el cálculo de cuotas y ratios financieros DTI, DSCR."""
        self.assertIn("cuota_mensual", self.df_enriched.columns)
        self.assertIn("ratio_cuota_ingreso", self.df_enriched.columns)
        self.assertIn("ratio_cobertura_deuda", self.df_enriched.columns)
        self.assertIn("score_comportamiento_alternativo", self.df_enriched.columns)

        # La cuota debe ser positiva
        self.assertTrue((self.df_enriched["cuota_mensual"] > 0).all())
        # DTI debe estar acotado
        self.assertTrue((self.df_enriched["ratio_cuota_ingreso"] > 0).all())

    def test_preprocessor_fit_transform(self):
        """Verifica el ajuste y transformación del preprocesador."""
        preprocessor = CreditDataPreprocessor()
        X_trans = preprocessor.fit_transform(self.df_enriched)
        
        self.assertEqual(X_trans.shape[0], 200)
        self.assertGreater(X_trans.shape[1], 20)
        self.assertFalse(np.isnan(X_trans).any())

    def test_model_training_and_inference(self):
        """Verifica el entrenamiento y predicción probabilística del modelo."""
        preprocessor = CreditDataPreprocessor()
        X_trans = preprocessor.fit_transform(self.df_enriched)
        y = self.df_enriched[config.TARGET_COLUMN].values

        model = CreditRiskXGBoostWrapper()
        model.fit(X_trans, y, feature_names=preprocessor.get_feature_names_out())
        
        probs = model.predict_proba(X_trans)
        self.assertEqual(probs.shape, (200, 2))
        self.assertTrue((probs[:, 1] >= 0.0).all() and (probs[:, 1] <= 1.0).all())

    def test_metrics_and_financial_calculations(self):
        """Verifica métricas de clasificación y pérdida esperada."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.75])
        
        metrics = compute_classification_metrics(y_true, y_prob)
        self.assertGreaterEqual(metrics["auc_roc"], 0.90)
        
        el = calculate_expected_loss(0.20, 10000.0, 0.50)
        self.assertEqual(el, 1000.0)

        score = convert_pd_to_credit_score(0.20)
        self.assertEqual(score, 800)

        decision, badge, desc = get_credit_decision(0.10)
        self.assertEqual(decision, "APROBADO")

    def test_xai_explainer_local_explanation(self):
        """Verifica la generación de explicaciones locales y narrativa del comité."""
        preprocessor = CreditDataPreprocessor()
        preprocessor.fit(self.df_enriched)
        
        X_trans = preprocessor.transform(self.df_enriched)
        y = self.df_enriched[config.TARGET_COLUMN].values
        
        model = CreditRiskXGBoostWrapper()
        model.fit(X_trans, y, feature_names=preprocessor.get_feature_names_out())

        explainer = CreditRiskExplainer(model, preprocessor)
        sample_dict = self.df_sample.iloc[0].to_dict()
        
        result = explainer.explain_single_applicant(sample_dict)
        self.assertIn("score_crediticio", result)
        self.assertIn("pd_value", result)
        self.assertIn("narrativa_comite", result)
        self.assertIn("shap_features", result)


if __name__ == "__main__":
    unittest.main()
