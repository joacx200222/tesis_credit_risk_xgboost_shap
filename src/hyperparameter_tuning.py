"""
Descripción: Optimización de hiperparámetros con Optuna (búsqueda bayesiana TPE)
             y validación cruzada estratificada de 5 folds sobre el set de
             entrenamiento. Se optimizan los tres modelos (XGBoost, Random Forest
             y Regresión Logística) para que la comparación final sea justa.
             El set de prueba no participa en la búsqueda.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, Any
import sys

import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.data_preprocessing import CreditDataPreprocessor
from src.model_evaluation import compute_roc_curve_and_auc, compute_kolmogorov_smirnov


def cross_validate(model_factory: Callable[[], Any], X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Evalúa un modelo con validación cruzada estratificada de 5 folds.
    Devuelve el AUC y el KS promedio de los 5 folds de validación.
    """
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    aucs, kss = [], []
    for train_idx, valid_idx in skf.split(X, y):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        prob = model.predict_proba(X[valid_idx])[:, 1]
        aucs.append(compute_roc_curve_and_auc(y[valid_idx], prob)[0])
        kss.append(compute_kolmogorov_smirnov(y[valid_idx], prob)[0])
    return {"cv_auc": round(float(np.mean(aucs)), 4), "cv_ks_pct": round(float(np.mean(kss)), 2)}


def xgboost_factory(params: Dict[str, Any], scale_pos_weight: float) -> Callable[[], xgb.XGBClassifier]:
    base = {k: v for k, v in config.XGBOOST_DEFAULT_PARAMS.items() if k not in params}
    base["scale_pos_weight"] = scale_pos_weight
    return lambda: xgb.XGBClassifier(**base, **params)


def random_forest_factory(params: Dict[str, Any]) -> Callable[[], RandomForestClassifier]:
    base = {k: v for k, v in config.RF_DEFAULT_PARAMS.items() if k not in params}
    return lambda: RandomForestClassifier(**base, **params)


def logistic_factory(params: Dict[str, Any]) -> Callable[[], LogisticRegression]:
    base = {k: v for k, v in config.LOGREG_DEFAULT_PARAMS.items() if k not in params}
    return lambda: LogisticRegression(**base, **params)


def run_hyperparameter_search(n_trials: Dict[str, int] = None) -> Dict[str, Any]:
    """
    Ejecuta la búsqueda de hiperparámetros para los tres modelos y guarda
    los mejores parámetros en models/best_params.json.

    Métrica objetivo: AUC promedio de la validación cruzada. Se usa el AUC y no
    el KS porque el KS por deciles en folds de ~168 casos (≈42 morosos) es muy
    discreto y ruidoso; el KS se reporta junto al AUC y se evalúa en test.
    """
    n_trials = n_trials or config.OPTUNA_TRIALS
    print("--- [Optimización de hiperparámetros: Optuna + validación cruzada estratificada 5-fold] ---")

    df_train = pd.read_csv(config.PROCESSED_TRAIN_PATH)
    preprocessor = CreditDataPreprocessor.load(config.PREPROCESSOR_PATH)
    X = preprocessor.transform(df_train)
    y = df_train[config.TARGET_COLUMN].values

    n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    scale_pos_weight = n_neg / n_pos
    print(f"Train: {X.shape} | {n_pos} morosos / {n_neg} cumplidores | scale_pos_weight = {scale_pos_weight:.2f}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = lambda: optuna.samplers.TPESampler(seed=config.RANDOM_STATE)

    # Espacios de búsqueda por modelo
    def xgb_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=20),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }

    def rf_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        }

    def lr_space(trial):
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}

    searches = {
        "XGBoost": (xgb_space, lambda p: xgboost_factory(p, scale_pos_weight)),
        "Random_Forest": (rf_space, random_forest_factory),
        "Regresion_Logistica": (lr_space, logistic_factory),
    }

    results = {}
    for name, (space, factory) in searches.items():
        # Línea base: los hiperparámetros fijos actuales, evaluados con la misma validación cruzada
        baseline = cross_validate(factory({}), X, y)

        def objective(trial):
            scores = cross_validate(factory(space(trial)), X, y)
            trial.set_user_attr("cv_ks_pct", scores["cv_ks_pct"])
            return scores["cv_auc"]

        study = optuna.create_study(direction="maximize", sampler=sampler())
        study.optimize(objective, n_trials=n_trials[name])

        best = study.best_trial
        results[name] = {
            "best_params": best.params,
            "cv_auc": round(best.value, 4),
            "cv_ks_pct": best.user_attrs["cv_ks_pct"],
            "baseline_cv_auc": baseline["cv_auc"],
            "baseline_cv_ks_pct": baseline["cv_ks_pct"],
            "n_trials": n_trials[name],
        }
        print(f"{name:<20} | CV AUC base {baseline['cv_auc']:.4f} -> optimizado {best.value:.4f} "
              f"| CV KS base {baseline['cv_ks_pct']:.2f}% -> {best.user_attrs['cv_ks_pct']:.2f}% "
              f"| {n_trials[name]} intentos")

    output = {
        "metodo": "Optuna TPE + validación cruzada estratificada de 5 folds sobre train",
        "metrica_objetivo": "AUC promedio de validación cruzada",
        "random_state": config.RANDOM_STATE,
        "modelos": results,
    }
    with open(config.BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[OK] Mejores hiperparámetros guardados en: {config.BEST_PARAMS_PATH}")
    return output


def load_best_params(model_name: str) -> Dict[str, Any]:
    """Devuelve los mejores hiperparámetros de un modelo, o {} si no se ha corrido la búsqueda."""
    if not config.BEST_PARAMS_PATH.exists():
        return {}
    with open(config.BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["modelos"].get(model_name, {}).get("best_params", {})


if __name__ == "__main__":
    run_hyperparameter_search()
