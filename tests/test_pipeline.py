import unittest
import numpy as np
import pandas as pd

from src import config
from src.ingenieria_variables import cargar_encuesta
from src.preprocesamiento import Preprocesador
from src.entrenamiento import ModeloXGBoost
from src.evaluacion import (
    calcular_metricas,
    perdida_esperada,
    pd_a_score,
    decision_crediticia,
)
from src.explicabilidad import cargar_explicador


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = cargar_encuesta(config.RUTA_DTA)

    def test_estructura_encuesta(self):
        self.assertEqual(len(self.df), 1201)
        self.assertIn(config.COLUMNA_OBJETIVO, self.df.columns)
        self.assertIn("adopcion_yape_plin", self.df.columns)
        self.assertIn("frecuencia_yape_plin", self.df.columns)
        self.assertIn("puntualidad_pago_deudas", self.df.columns)
        self.assertIn("guarda_dinero_recibos", self.df.columns)

        tasa_mora = self.df[config.COLUMNA_OBJETIVO].mean()
        self.assertAlmostEqual(tasa_mora, 0.2506, delta=0.01)

    def test_indices_entre_0_y_1(self):
        for col in config.INDICES:
            self.assertIn(col, self.df.columns)
            self.assertTrue((self.df[col] >= 0.0).all())
            self.assertTrue((self.df[col] <= 1.0).all())

    def test_preprocesador(self):
        preprocesador = Preprocesador()
        X_t = preprocesador.fit_transform(self.df)

        self.assertEqual(X_t.shape[0], 1201)
        self.assertGreater(X_t.shape[1], 50)
        self.assertFalse(np.isnan(X_t).any())
        self.assertEqual(len(preprocesador.get_feature_names_out()), X_t.shape[1])

    def test_entrenamiento_y_prediccion(self):
        preprocesador = Preprocesador()
        X_t = preprocesador.fit_transform(self.df)
        y = self.df[config.COLUMNA_OBJETIVO].values

        modelo = ModeloXGBoost()
        modelo.fit(X_t, y, nombres_variables=preprocesador.get_feature_names_out())

        probs = modelo.predict_proba(X_t)
        self.assertEqual(probs.shape, (1201, 2))
        self.assertTrue((probs[:, 1] >= 0.0).all() and (probs[:, 1] <= 1.0).all())

    def test_metricas_y_calculos(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.75])

        metricas = calcular_metricas(y_true, y_prob)
        self.assertGreaterEqual(metricas["auc_roc"], 0.90)

        el = perdida_esperada(0.20, 10000.0, 0.50)
        self.assertEqual(el, 1000.0)

        score = pd_a_score(0.20)
        self.assertEqual(score, 800)

        decision, badge, desc = decision_crediticia(0.15)
        self.assertEqual(decision, "APROBADO")

        decision_med, _, _ = decision_crediticia(0.35)
        self.assertEqual(decision_med, "REVISIÓN MANUAL")

        decision_high, _, _ = decision_crediticia(0.60)
        self.assertEqual(decision_high, "RECHAZADO")

    def test_explicacion_local_y_motivos(self):
        explicador = cargar_explicador()
        solicitud = self.df.iloc[0].to_dict()

        result = explicador.evaluar_solicitud(solicitud)
        self.assertIn("score", result)
        self.assertIn("pd", result)
        self.assertIn("dictamen", result)
        self.assertIn("contribuciones", result)
        self.assertIn("motivos_denegacion", result)
        self.assertTrue(len(result["motivos_denegacion"]) >= 1)

    def test_aditividad_shap(self):
        # E[f(x)] + sum(phi) tiene que dar el margen del modelo
        explicador = cargar_explicador()
        df_test = pd.read_csv(config.RUTA_TEST)
        X_test = explicador.preprocesador.transform(df_test)

        matriz_shap = explicador.shap_local(X_test)
        margen = explicador.modelo.xgb.predict(X_test, output_margin=True)
        error = np.abs(explicador.valor_base + matriz_shap.sum(axis=1) - margen).max()
        self.assertLess(error, 1e-4)

    def test_umbrales_politica(self):
        self.assertEqual(decision_crediticia(0.30)[0], "APROBADO")
        self.assertEqual(decision_crediticia(0.45)[0], "REVISIÓN MANUAL")
        self.assertEqual(decision_crediticia(0.4501)[0], "RECHAZADO")

    def test_perfiles_ejemplo_dan_su_decision(self):
        from src.app import perfiles_ejemplo
        explicador = cargar_explicador()
        esperado = {"aprobado": "APROBADO", "revisión manual": "REVISIÓN MANUAL", "rechazado": "RECHAZADO"}
        for nombre, perfil in perfiles_ejemplo().items():
            for etiqueta, decision in esperado.items():
                if etiqueta in nombre:
                    result = explicador.evaluar_solicitud(perfil)
                    self.assertEqual(result["decision"], decision, msg=nombre)


if __name__ == "__main__":
    unittest.main()
