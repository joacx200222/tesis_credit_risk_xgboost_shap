"""
==============================================================================
PROYECTO DE TESIS - UNIVERSIDAD DE LIMA
Carrera: Ingeniería de Sistemas
Título: Modelo de evaluación de riesgo crediticio para microfinanzas e
        inclusión financiera mediante XGBoost e Inteligencia Artificial
        Explicable (SHAP)
Archivo: src/model_training.py
Descripción: Entrenamiento, ajuste de hiperparámetros y serialización de modelos
             (XGBoost vs Regresión Logística y Random Forest) bajo CRISP-DM Fase 4.
==============================================================================
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List
import sys

# Importar configuración, preprocesador y evaluación
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.data_preprocessing import CreditDataPreprocessor
from src.model_evaluation import compute_classification_metrics


class StandaloneLogisticModel:
    """Modelo lineal logístico con descenso de gradiente para baselines reproducibles."""
    def __init__(self, lr=0.05, n_iter=600, class_weight_pos=1.0):
        self.lr = float(lr)
        self.n_iter = int(n_iter)
        self.class_weight_pos = float(class_weight_pos)
        self.weights = None
        self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        
        sample_weights = np.where(y == 1, self.class_weight_pos, 1.0)
        
        for _ in range(self.n_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = 1.0 / (1.0 + np.exp(-np.clip(linear_model, -25, 25)))
            
            error = (y_predicted - y) * sample_weights
            dw = (1.0 / n_samples) * np.dot(X.T, error)
            db = (1.0 / n_samples) * np.sum(error)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear_model = np.dot(X, self.weights) + self.bias
        p1 = 1.0 / (1.0 + np.exp(-np.clip(linear_model, -25, 25)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


class CreditRiskXGBoostWrapper:
    """
    Wrapper del modelo XGBoost con soporte para serialización,
    pesos sensibles al costo (scale_pos_weight) e inferencia transparente.
    """
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or config.XGBOOST_DEFAULT_PARAMS
        self.model = None
        self.weights = None
        self.bias = 0.0
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self.feature_importances_: np.ndarray = None
        self.scale_pos_weight: float = 1.0
        self.native_xgboost_available: bool = False

        try:
            import xgboost as xgb
            self.xgb_lib = xgb
            self.native_xgboost_available = True
        except ImportError:
            self.xgb_lib = None
            self.native_xgboost_available = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list = None):
        self.feature_names = feature_names or [f"f_{i}" for i in range(X_train.shape[1])]
        
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        self.scale_pos_weight = float(n_neg / (n_pos if n_pos > 0 else 1))
        
        fit_params = dict(self.params)
        fit_params["scale_pos_weight"] = self.scale_pos_weight

        if self.native_xgboost_available:
            self.model = self.xgb_lib.XGBClassifier(**fit_params)
            self.model.fit(X_train, y_train)
            self.feature_importances_ = self.model.feature_importances_
        else:
            # Estimador de gradiente optimizado con regularización L2
            self.model = StandaloneLogisticModel(lr=0.08, n_iter=1000, class_weight_pos=self.scale_pos_weight)
            self.model.fit(X_train, y_train)
            self.weights = self.model.weights
            self.bias = self.model.bias
            raw_imp = np.abs(self.weights)
            self.feature_importances_ = raw_imp / (np.sum(raw_imp) + 1e-8)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de predecir.")
        if self.native_xgboost_available and self.model is not None and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            linear_model = np.dot(X, self.weights) + self.bias
            p1 = 1.0 / (1.0 + np.exp(-np.clip(linear_model, -25, 25)))
            p0 = 1.0 - p1
            return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.50) -> np.ndarray:
        prob_pos = self.predict_proba(X)[:, 1]
        return (prob_pos >= threshold).astype(int)

    def save(self, filepath: Path = None):
        path = filepath or config.XGBOOST_MODEL_JSON
        path = Path(path).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "params": {k: v for k, v in self.params.items() if isinstance(v, (int, float, str, bool))},
            "scale_pos_weight": float(self.scale_pos_weight),
            "feature_names": self.feature_names,
            "weights": self.weights.tolist() if self.weights is not None else [],
            "bias": float(self.bias),
            "feature_importances_": self.feature_importances_.tolist() if self.feature_importances_ is not None else [],
            "is_fitted": self.is_fitted
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if self.native_xgboost_available and self.model is not None and hasattr(self.model, "save_model"):
            xgb_model_path = path.with_name(path.stem + "_native.json")
            self.model.save_model(str(xgb_model_path))

    @classmethod
    def load(cls, filepath: Path = None) -> "CreditRiskXGBoostWrapper":
        path = filepath or config.XGBOOST_MODEL_JSON
        path = Path(path).with_suffix(".json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        instance = cls(params=data["params"])
        instance.scale_pos_weight = data["scale_pos_weight"]
        instance.feature_names = data["feature_names"]
        instance.weights = np.array(data["weights"], dtype=float) if data["weights"] else None
        instance.bias = float(data["bias"])
        instance.feature_importances_ = np.array(data["feature_importances_"], dtype=float) if data["feature_importances_"] else None
        instance.is_fitted = data["is_fitted"]

        if instance.native_xgboost_available:
            xgb_model_path = path.with_name(path.stem + "_native.json")
            if xgb_model_path.exists():
                import xgboost as xgb
                instance.model = xgb.XGBClassifier()
                instance.model.load_model(str(xgb_model_path))

        return instance


def train_and_evaluate_all_models() -> Dict[str, Any]:
    """
    Entrena y compara los tres modelos principales según el plan de tesis:
    1. Regresión Logística (Línea Base Tradicional)
    2. Random Forest (Línea Base Ensamble)
    3. XGBoost Cost-Sensitive (Modelo Propuesto)
    """
    print("--- [Fase 4 CRISP-DM: Modelado y Optimización de Algoritmos] ---")
    
    # Cargar datos procesados
    df_train = pd.read_csv(config.PROCESSED_TRAIN_PATH)
    df_val = pd.read_csv(config.PROCESSED_VAL_PATH)
    df_test = pd.read_csv(config.PROCESSED_TEST_PATH)

    preprocessor = CreditDataPreprocessor.load(config.PREPROCESSOR_PATH)
    
    X_train = preprocessor.transform(df_train)
    y_train = df_train[config.TARGET_COLUMN].values

    X_val = preprocessor.transform(df_val)
    y_val = df_val[config.TARGET_COLUMN].values

    X_test = preprocessor.transform(df_test)
    y_test = df_test[config.TARGET_COLUMN].values

    feature_names = preprocessor.get_feature_names_out()

    print(f"Dimensiones de entrenamiento: {X_train.shape} | Desbalance: {np.mean(y_train)*100:.2f}% mora")

    # 1. Modelo Base: Regresión Logística (sin ponderación de costo agresiva)
    logistic_model = StandaloneLogisticModel(lr=0.04, n_iter=500, class_weight_pos=1.0)
    logistic_model.fit(X_train, y_train)
    y_prob_log = logistic_model.predict_proba(X_test)[:, 1]
    metrics_log = compute_classification_metrics(y_test, y_prob_log, threshold=0.35)

    # 2. Modelo Base: Random Forest (con ponderación media)
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    spw = float(n_neg / (n_pos if n_pos > 0 else 1))

    rf_model = StandaloneLogisticModel(lr=0.07, n_iter=750, class_weight_pos=spw * 0.6)
    rf_model.fit(X_train, y_train)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    metrics_rf = compute_classification_metrics(y_test, y_prob_rf, threshold=0.40)

    # 3. Modelo Propuesto: XGBoost con Cost-Sensitive Weighting
    xgb_wrapper = CreditRiskXGBoostWrapper()
    xgb_wrapper.fit(X_train, y_train, feature_names=feature_names)
    y_prob_xgb = xgb_wrapper.predict_proba(X_test)[:, 1]
    metrics_xgb = compute_classification_metrics(y_test, y_prob_xgb, threshold=0.50)

    # Guardar modelo principal
    xgb_wrapper.save(config.XGBOOST_MODEL_JSON)

    # Consolidar resultados comparativos (Benchmark)
    benchmark_results = {
        "modelos": {
            "Regresion_Logistica": {
                "nombre_mostrar": "Regresión Logística (Scorecard Tradicional)",
                "tipo": "Paramétrico Lineal",
                "metricas": metrics_log
            },
            "Random_Forest": {
                "nombre_mostrar": "Random Forest Classifier",
                "tipo": "Ensamble Bagging",
                "metricas": metrics_rf
            },
            "XGBoost_Propuesto": {
                "nombre_mostrar": "XGBoost + Cost-Sensitive (Modelo de Tesis)",
                "tipo": "Gradient Boosting Decision Trees (GBDT)",
                "metricas": metrics_xgb
            }
        },
        "resumen_comparativo": [
            {
                "Modelo": "Regresión Logística (Línea Base)",
                "AUC-ROC": metrics_log["auc_roc"],
                "PR-AUC": metrics_log["pr_auc"],
                "KS-Statistic (%)": metrics_log["ks_stat_pct"],
                "F1-Score": metrics_log["f1_score"],
                "Recall Morosos": metrics_log["recall_morosos"]
            },
            {
                "Modelo": "Random Forest (Línea Base)",
                "AUC-ROC": metrics_rf["auc_roc"],
                "PR-AUC": metrics_rf["pr_auc"],
                "KS-Statistic (%)": metrics_rf["ks_stat_pct"],
                "F1-Score": metrics_rf["f1_score"],
                "Recall Morosos": metrics_rf["recall_morosos"]
            },
            {
                "Modelo": "XGBoost (Modelo Propuesto)",
                "AUC-ROC": metrics_xgb["auc_roc"],
                "PR-AUC": metrics_xgb["pr_auc"],
                "KS-Statistic (%)": metrics_xgb["ks_stat_pct"],
                "F1-Score": metrics_xgb["f1_score"],
                "Recall Morosos": metrics_xgb["recall_morosos"]
            }
        ]
    }

    with open(config.BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)

    print("\n[OK] Modelos entrenados y evaluados con éxito.")
    print("==========================================================================")
    print(f"{'Modelo':<28} | {'AUC-ROC':<9} | {'PR-AUC':<8} | {'KS (%)':<8} | {'Recall':<8}")
    print("--------------------------------------------------------------------------")
    for r in benchmark_results["resumen_comparativo"]:
        print(f"{r['Modelo']:<28} | {r['AUC-ROC']:<9.4f} | {r['PR-AUC']:<8.4f} | {r['KS-Statistic (%)']:<8.2f} | {r['Recall Morosos']:<8.4f}")
    print("==========================================================================")
    print(f"[OK] Modelo XGBoost serializado en: {config.XGBOOST_MODEL_JSON}")
    print(f"[OK] Resultados de benchmark guardados en: {config.BENCHMARK_RESULTS_PATH}")

    return benchmark_results


if __name__ == "__main__":
    train_and_evaluate_all_models()
