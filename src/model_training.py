"""
Descripción: Entrenamiento, ajuste de hiperparámetros y serialización de modelos
             (XGBoost vs Regresión Logística y Random Forest) con la base SBS 2022.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Importar configuración, preprocesador y evaluación
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.data_preprocessing import CreditDataPreprocessor
from src.model_evaluation import compute_classification_metrics


class CreditRiskXGBoostWrapper:
    """
    Wrapper del modelo XGBoost con soporte para serialización en .joblib y .json,
    pesos sensibles al costo (scale_pos_weight = 2.99) e inferencia transparente.
    """
    def __init__(self, params: Dict[str, Any] = None):
        self.params = dict(params or config.XGBOOST_DEFAULT_PARAMS)
        self.model: xgb.XGBClassifier = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self.feature_importances_: np.ndarray = None
        self.scale_pos_weight: float = float(config.SCALE_POS_WEIGHT)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, feature_names: list = None):
        self.feature_names = feature_names or [f"f_{i}" for i in range(X_train.shape[1])]
        
        # Calcular el desbalance real en Train
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        self.scale_pos_weight = float(n_neg / (n_pos if n_pos > 0 else 1))
        
        fit_params = dict(self.params)
        fit_params["scale_pos_weight"] = self.scale_pos_weight

        self.model = xgb.XGBClassifier(**fit_params)

        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)

        self.feature_importances_ = self.model.feature_importances_
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir.")
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.50) -> np.ndarray:
        prob_pos = self.predict_proba(X)[:, 1]
        return (prob_pos >= threshold).astype(int)

    def save(self, filepath: Path = None):
        path = filepath or config.XGBOOST_MODEL_JSON
        path = Path(path).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar en JSON (metadatos e hiperparámetros)
        data = {
            "params": {k: v for k, v in self.params.items() if isinstance(v, (int, float, str, bool))},
            "scale_pos_weight": float(self.scale_pos_weight),
            "feature_names": self.feature_names,
            "feature_importances_": self.feature_importances_.tolist() if self.feature_importances_ is not None else [],
            "is_fitted": self.is_fitted
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Guardar modelo nativo XGBoost
        xgb_model_path = path.with_name(path.stem + "_native.json")
        self.model.save_model(str(xgb_model_path))

        # Guardar con joblib (modelo principal y alias model.joblib)
        joblib_path = path.with_suffix(".joblib")
        joblib.dump(self, joblib_path)
        
        # Guardar también como model.joblib en models/ para facilitar consumo
        alias_joblib = path.parent / "model.joblib"
        joblib.dump(self, alias_joblib)

    @classmethod
    def load(cls, filepath: Path = None) -> "CreditRiskXGBoostWrapper":
        path = filepath or config.XGBOOST_MODEL_JSON
        path = Path(path)
        
        # Si se pasa un .joblib o existe .joblib, cargarlo directamente
        joblib_path = path.with_suffix(".joblib")
        if joblib_path.exists():
            try:
                loaded = joblib.load(joblib_path)
                if isinstance(loaded, cls):
                    return loaded
            except Exception:
                pass

        json_path = path.with_suffix(".json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        instance = cls(params=data["params"])
        instance.scale_pos_weight = data["scale_pos_weight"]
        instance.feature_names = data["feature_names"]
        instance.feature_importances_ = np.array(data["feature_importances_"], dtype=float) if data["feature_importances_"] else None
        instance.is_fitted = data["is_fitted"]

        xgb_model_path = json_path.with_name(json_path.stem + "_native.json")
        if xgb_model_path.exists():
            instance.model = xgb.XGBClassifier()
            instance.model.load_model(str(xgb_model_path))

        return instance


def train_and_evaluate_all_models() -> Dict[str, Any]:
    """
    Entrena y compara los tres modelos principales según el plan de tesis:
    1. Regresión Logística (Línea Base Tradicional)
    2. Random Forest (Línea Base Ensamble)
    3. XGBoost Cost-Sensitive (Modelo Propuesto con Base Empírica SBS 2022)
    """
    print("--- [Fase 4 CRISP-DM: Modelado y Optimización con Base SBS/CAF 2022] ---")
    
    # Cargar datasets particionados
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
    print(f"Dimensiones de prueba:        {X_test.shape}  | Desbalance: {np.mean(y_test)*100:.2f}% mora")

    # 1. Modelo Base: Regresión Logística con balanceo de clases
    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE)
    log_reg.fit(X_train, y_train)
    y_prob_log = log_reg.predict_proba(X_test)[:, 1]
    metrics_log = compute_classification_metrics(y_test, y_prob_log, threshold=0.50)

    # 2. Modelo Base: Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_split=5,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)
    y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]
    metrics_rf = compute_classification_metrics(y_test, y_prob_rf, threshold=0.50)

    # 3. Modelo Propuesto: XGBoost con Cost-Sensitive Weighting y regularización gamma/lambda
    xgb_wrapper = CreditRiskXGBoostWrapper()
    xgb_wrapper.fit(X_train, y_train, X_val=X_val, y_val=y_val, feature_names=feature_names)
    y_prob_xgb = xgb_wrapper.predict_proba(X_test)[:, 1]
    metrics_xgb = compute_classification_metrics(y_test, y_prob_xgb, threshold=0.50)

    # Guardar modelo principal serializado
    xgb_wrapper.save(config.XGBOOST_MODEL_JSON)
    # También serializar preprocesador en formato joblib
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH.with_suffix(".joblib"))

    # Consolidar resultados comparativos (Benchmark institucional SBS)
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
                "Modelo": "XGBoost (Modelo Propuesto SBS)",
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
    print(f"{'Modelo':<30} | {'AUC-ROC':<9} | {'PR-AUC':<8} | {'KS (%)':<8} | {'Recall':<8}")
    print("--------------------------------------------------------------------------")
    for r in benchmark_results["resumen_comparativo"]:
        print(f"{r['Modelo']:<30} | {r['AUC-ROC']:<9.4f} | {r['PR-AUC']:<8.4f} | {r['KS-Statistic (%)']:<8.2f} | {r['Recall Morosos']:<8.4f}")
    print("==========================================================================")
    print(f"[OK] Modelo XGBoost serializado en:")
    print(f"     - {config.XGBOOST_MODEL_JSON}")
    print(f"     - {config.XGBOOST_MODEL_PATH}")
    print(f"     - {config.MODELS_DIR / 'model.joblib'}")
    print(f"[OK] Resultados de benchmark guardados en: {config.BENCHMARK_RESULTS_PATH}")

    return benchmark_results


if __name__ == "__main__":
    train_and_evaluate_all_models()
