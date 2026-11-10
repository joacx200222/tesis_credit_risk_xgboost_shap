import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.linear_model import LogisticRegression

from src import config
from src import busqueda_hiperparametros as bh


class TestBusquedaHiperparametros(unittest.TestCase):

    def test_validacion_cruzada_en_rango(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4))
        y = (X[:, 0] + rng.normal(scale=0.5, size=200) > 0.5).astype(int)

        metricas = bh.validacion_cruzada(lambda: LogisticRegression(), X, y)

        self.assertGreater(metricas["cv_auc"], 0.8)  # X[:, 0] separa bien las clases
        self.assertLessEqual(metricas["cv_auc"], 1.0)
        self.assertTrue(0.0 <= metricas["cv_ks_pct"] <= 100.0)

    def test_crear_modelos_respeta_params(self):
        modelo = bh.crear_xgboost({"max_depth": 2}, scale_pos_weight=3.0)()
        self.assertEqual(modelo.get_params()["max_depth"], 2)
        self.assertEqual(modelo.get_params()["scale_pos_weight"], 3.0)
        # lo que no se optimiza sale de los params base
        self.assertEqual(modelo.get_params()["random_state"], config.SEMILLA)

        rf = bh.crear_random_forest({"n_estimators": 10})()
        self.assertEqual(rf.n_estimators, 10)
        self.assertEqual(rf.class_weight, "balanced")

    def test_sin_archivo_devuelve_vacio(self):
        with mock.patch.object(config, "RUTA_MEJORES_PARAMS", Path(tempfile.gettempdir()) / "no_existe_best_params.json"):
            self.assertEqual(bh.cargar_mejores_params("XGBoost"), {})

    def test_busqueda_corta_guarda_y_se_puede_recargar(self):
        intentos_prueba = {"XGBoost": 2, "Random_Forest": 2, "Regresion_Logistica": 2}
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "mejores_params.json"
            with mock.patch.object(config, "RUTA_MEJORES_PARAMS", ruta):
                salida = bh.buscar_hiperparametros(intentos=intentos_prueba)

                self.assertTrue(ruta.exists())
                with open(ruta, encoding="utf-8") as f:
                    self.assertEqual(json.load(f), salida)
                for nombre in intentos_prueba:
                    self.assertEqual(salida["modelos"][nombre]["intentos"], 2)
                    self.assertEqual(bh.cargar_mejores_params(nombre), salida["modelos"][nombre]["mejores_params"])


if __name__ == "__main__":
    unittest.main()
