import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.linear_model import LogisticRegression

from src import config
from src import hyperparameter_tuning as ht


class TestHyperparameterTuning(unittest.TestCase):

    def test_cross_validate_devuelve_promedios_en_rango(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4))
        y = (X[:, 0] + rng.normal(scale=0.5, size=200) > 0.5).astype(int)

        scores = ht.cross_validate(lambda: LogisticRegression(), X, y)

        self.assertGreater(scores["cv_auc"], 0.8)  # X[:, 0] separa bien las clases
        self.assertLessEqual(scores["cv_auc"], 1.0)
        self.assertTrue(0.0 <= scores["cv_ks_pct"] <= 100.0)

    def test_factories_respetan_los_params_que_se_pasan(self):
        model = ht.xgboost_factory({"max_depth": 2}, scale_pos_weight=3.0)()
        self.assertEqual(model.get_params()["max_depth"], 2)
        self.assertEqual(model.get_params()["scale_pos_weight"], 3.0)
        # lo que no se optimiza sale de los params base
        self.assertEqual(model.get_params()["random_state"], config.RANDOM_STATE)

        rf = ht.random_forest_factory({"n_estimators": 10})()
        self.assertEqual(rf.n_estimators, 10)
        self.assertEqual(rf.class_weight, "balanced")

    def test_load_best_params_sin_archivo_devuelve_vacio(self):
        with mock.patch.object(config, "BEST_PARAMS_PATH", Path(tempfile.gettempdir()) / "no_existe_best_params.json"):
            self.assertEqual(ht.load_best_params("XGBoost"), {})

    def test_busqueda_corta_guarda_y_se_puede_recargar(self):
        trials = {"XGBoost": 2, "Random_Forest": 2, "Regresion_Logistica": 2}
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "best_params.json"
            with mock.patch.object(config, "BEST_PARAMS_PATH", out_path):
                output = ht.run_hyperparameter_search(n_trials=trials)

                self.assertTrue(out_path.exists())
                with open(out_path, encoding="utf-8") as f:
                    self.assertEqual(json.load(f), output)
                for nombre in trials:
                    self.assertEqual(output["modelos"][nombre]["n_trials"], 2)
                    self.assertEqual(ht.load_best_params(nombre), output["modelos"][nombre]["best_params"])


if __name__ == "__main__":
    unittest.main()
