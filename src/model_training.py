import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from src import config
from src.data_preprocessing import CreditDataPreprocessor
from src.model_evaluation import compute_classification_metrics
from src.hyperparameter_tuning import load_best_params


class CreditRiskXGBoostWrapper:
    """
    XGBClassifier con scale_pos_weight calculado del train (negativos / positivos),
    guardado en dos JSON: metadatos y modelo nativo.
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

        data = {
            "params": {k: v for k, v in self.params.items() if isinstance(v, (int, float, str, bool))},
            "scale_pos_weight": float(self.scale_pos_weight),
            "feature_names": self.feature_names,
            "feature_importances_": self.feature_importances_.tolist() if self.feature_importances_ is not None else [],
            "is_fitted": self.is_fitted
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        xgb_model_path = path.with_name(path.stem + "_native.json")
        self.model.save_model(str(xgb_model_path))

    @classmethod
    def load(cls, filepath: Path = None) -> "CreditRiskXGBoostWrapper":
        json_path = Path(filepath or config.XGBOOST_MODEL_JSON).with_suffix(".json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls(params=data["params"])
        instance.scale_pos_weight = data["scale_pos_weight"]
        instance.feature_names = data["feature_names"]
        instance.feature_importances_ = np.array(data["feature_importances_"], dtype=float) if data["feature_importances_"] else None
        instance.is_fitted = data["is_fitted"]

        xgb_model_path = json_path.with_name(json_path.stem + "_native.json")
        if not xgb_model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo nativo de XGBoost en: {xgb_model_path}")
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(str(xgb_model_path))

        return instance


def train_and_evaluate_all_models() -> Dict[str, Any]:
    """Entrena regresión logística, random forest y XGBoost (base y optimizados) y los compara en test."""

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

    print(f"train {X_train.shape}, mora {np.mean(y_train)*100:.2f}% | test {X_test.shape}, mora {np.mean(y_test)*100:.2f}%")

    # {} si todavía no se corrió hyperparameter_tuning
    best_log = load_best_params("Regresion_Logistica")
    best_rf = load_best_params("Random_Forest")
    best_xgb = load_best_params("XGBoost")
    optimizado = bool(best_log and best_rf and best_xgb)
    print(f"Hiperparámetros: {'optimizados con Optuna' if optimizado else 'base (sin optimizar)'}")

    def evaluar(modelo):
        return compute_classification_metrics(y_test, modelo.predict_proba(X_test)[:, 1], threshold=0.50)

    log_reg_base = LogisticRegression(**config.LOGREG_DEFAULT_PARAMS).fit(X_train, y_train)
    log_reg = LogisticRegression(**{**config.LOGREG_DEFAULT_PARAMS, **best_log}).fit(X_train, y_train)
    metrics_log_base, metrics_log = evaluar(log_reg_base), evaluar(log_reg)

    rf_base = RandomForestClassifier(**config.RF_DEFAULT_PARAMS).fit(X_train, y_train)
    rf_clf = RandomForestClassifier(**{**config.RF_DEFAULT_PARAMS, **best_rf}).fit(X_train, y_train)
    metrics_rf_base, metrics_rf = evaluar(rf_base), evaluar(rf_clf)

    xgb_base = CreditRiskXGBoostWrapper()
    xgb_base.fit(X_train, y_train, X_val=X_val, y_val=y_val, feature_names=feature_names)
    xgb_wrapper = CreditRiskXGBoostWrapper(params={**config.XGBOOST_DEFAULT_PARAMS, **best_xgb})
    xgb_wrapper.fit(X_train, y_train, X_val=X_val, y_val=y_val, feature_names=feature_names)
    metrics_xgb_base, metrics_xgb = evaluar(xgb_base), evaluar(xgb_wrapper)

    xgb_wrapper.save(config.XGBOOST_MODEL_JSON)

    benchmark_results = {
        "hiperparametros_optimizados": optimizado,
        "comparacion_base_vs_optimizado": [
            {
                "Modelo": nombre,
                "KS base (%)": base["ks_stat_pct"],
                "KS optimizado (%)": opt["ks_stat_pct"],
                "AUC base": base["auc_roc"],
                "AUC optimizado": opt["auc_roc"],
            }
            for nombre, base, opt in [
                ("Regresión Logística", metrics_log_base, metrics_log),
                ("Random Forest", metrics_rf_base, metrics_rf),
                ("XGBoost", metrics_xgb_base, metrics_xgb),
            ]
        ],
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

    print()
    print(f"{'Modelo':<30} | {'AUC-ROC':<9} | {'PR-AUC':<8} | {'KS (%)':<8} | {'Recall':<8}")
    for r in benchmark_results["resumen_comparativo"]:
        print(f"{r['Modelo']:<30} | {r['AUC-ROC']:<9.4f} | {r['PR-AUC']:<8.4f} | {r['KS-Statistic (%)']:<8.2f} | {r['Recall Morosos']:<8.4f}")
    print()
    print(f"{'Base vs optimizado (test)':<30} | {'KS base':<8} | {'KS opt':<8} | {'AUC base':<8} | {'AUC opt':<8}")
    for r in benchmark_results["comparacion_base_vs_optimizado"]:
        print(f"{r['Modelo']:<30} | {r['KS base (%)']:<8.2f} | {r['KS optimizado (%)']:<8.2f} | {r['AUC base']:<8.4f} | {r['AUC optimizado']:<8.4f}")

    return benchmark_results


if __name__ == "__main__":
    train_and_evaluate_all_models()
