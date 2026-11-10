import unittest
import numpy as np
import pandas as pd

from src import config
from src.feature_engineering import load_and_extract_sbs_dataset
from src.data_preprocessing import CreditDataPreprocessor
from src.model_training import CreditRiskXGBoostWrapper
from src.model_evaluation import (
    compute_classification_metrics,
    calculate_expected_loss,
    convert_pd_to_credit_score,
    get_credit_decision,
)
from src.explainability import get_default_explainer


class TestCreditRiskPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df_sbs = load_and_extract_sbs_dataset(config.RAW_DTA_PATH)

    def test_sbs_dataset_structure(self):
        self.assertEqual(len(self.df_sbs), 1201)
        self.assertIn(config.TARGET_COLUMN, self.df_sbs.columns)
        self.assertIn("adopcion_yape_plin", self.df_sbs.columns)
        self.assertIn("frecuencia_yape_plin", self.df_sbs.columns)
        self.assertIn("puntualidad_pago_deudas", self.df_sbs.columns)
        self.assertIn("guarda_dinero_recibos", self.df_sbs.columns)

        default_rate = self.df_sbs[config.TARGET_COLUMN].mean()
        self.assertAlmostEqual(default_rate, 0.2506, delta=0.01)

    def test_feature_engineering_indices(self):
        for col in config.NUMERICAL_DERIVED_FEATURES:
            self.assertIn(col, self.df_sbs.columns)
            self.assertTrue((self.df_sbs[col] >= 0.0).all())
            self.assertTrue((self.df_sbs[col] <= 1.0).all())

    def test_preprocessor_fit_transform(self):
        preprocessor = CreditDataPreprocessor()
        X_trans = preprocessor.fit_transform(self.df_sbs)

        self.assertEqual(X_trans.shape[0], 1201)
        self.assertGreater(X_trans.shape[1], 50)
        self.assertFalse(np.isnan(X_trans).any())
        self.assertEqual(len(preprocessor.get_feature_names_out()), X_trans.shape[1])

    def test_model_training_and_inference(self):
        preprocessor = CreditDataPreprocessor()
        X_trans = preprocessor.fit_transform(self.df_sbs)
        y = self.df_sbs[config.TARGET_COLUMN].values

        model = CreditRiskXGBoostWrapper()
        model.fit(X_trans, y, feature_names=preprocessor.get_feature_names_out())

        probs = model.predict_proba(X_trans)
        self.assertEqual(probs.shape, (1201, 2))
        self.assertTrue((probs[:, 1] >= 0.0).all() and (probs[:, 1] <= 1.0).all())

    def test_metrics_and_financial_calculations(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.75])

        metrics = compute_classification_metrics(y_true, y_prob)
        self.assertGreaterEqual(metrics["auc_roc"], 0.90)

        el = calculate_expected_loss(0.20, 10000.0, 0.50)
        self.assertEqual(el, 1000.0)

        score = convert_pd_to_credit_score(0.20)
        self.assertEqual(score, 800)

        decision, badge, desc = get_credit_decision(0.15)
        self.assertEqual(decision, "APROBADO")

        decision_med, _, _ = get_credit_decision(0.35)
        self.assertEqual(decision_med, "REVISIÓN MANUAL")

        decision_high, _, _ = get_credit_decision(0.60)
        self.assertEqual(decision_high, "RECHAZADO")

    def test_treeshap_explainer_local_and_adverse_actions(self):
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
        # E[f(x)] + sum(phi) tiene que dar el margen del modelo
        explainer = get_default_explainer()
        df_test = pd.read_csv(config.PROCESSED_TEST_PATH)
        X_test = explainer.preprocessor.transform(df_test)

        shap_matrix = explainer.compute_local_shap(X_test)
        margin = explainer.model_wrapper.model.predict(X_test, output_margin=True)
        error = np.abs(explainer.expected_value + shap_matrix.sum(axis=1) - margin).max()
        self.assertLess(error, 1e-4)

    def test_credit_policy_thresholds(self):
        self.assertEqual(get_credit_decision(0.30)[0], "APROBADO")
        self.assertEqual(get_credit_decision(0.45)[0], "REVISIÓN MANUAL")
        self.assertEqual(get_credit_decision(0.4501)[0], "RECHAZADO")

    def test_sample_profiles_match_their_label(self):
        from src.app import create_sample_profiles
        explainer = get_default_explainer()
        esperado = {"aprobado": "APROBADO", "revisión manual": "REVISIÓN MANUAL", "rechazado": "RECHAZADO"}
        for nombre, perfil in create_sample_profiles().items():
            for etiqueta, decision in esperado.items():
                if etiqueta in nombre:
                    result = explainer.explain_single_applicant(perfil)
                    self.assertEqual(result["decision"], decision, msg=nombre)


if __name__ == "__main__":
    unittest.main()
