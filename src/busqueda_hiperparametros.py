"""
Búsqueda de hiperparámetros con Optuna (TPE) y CV estratificada de 5 folds,
solo sobre train. Se optimizan los tres modelos para que la comparación sea
pareja; test no se toca acá.
"""

import json
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any

import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src import config
from src.preprocesamiento import Preprocesador
from src.evaluacion import curva_roc, estadistico_ks


def validacion_cruzada(crear_modelo: Callable[[], Any], X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """AUC y KS promedio de los folds de validación."""
    folds = StratifiedKFold(n_splits=config.FOLDS_CV, shuffle=True, random_state=config.SEMILLA)
    aucs, kss = [], []
    for idx_train, idx_valid in folds.split(X, y):
        modelo = crear_modelo()
        modelo.fit(X[idx_train], y[idx_train])
        prob = modelo.predict_proba(X[idx_valid])[:, 1]
        aucs.append(curva_roc(y[idx_valid], prob)[0])
        kss.append(estadistico_ks(y[idx_valid], prob)[0])
    return {"cv_auc": round(float(np.mean(aucs)), 4), "cv_ks_pct": round(float(np.mean(kss)), 2)}


def crear_xgboost(params: Dict[str, Any], scale_pos_weight: float) -> Callable[[], xgb.XGBClassifier]:
    base = {k: v for k, v in config.PARAMS_BASE_XGBOOST.items() if k not in params}
    base["scale_pos_weight"] = scale_pos_weight
    return lambda: xgb.XGBClassifier(**base, **params)


def crear_random_forest(params: Dict[str, Any]) -> Callable[[], RandomForestClassifier]:
    base = {k: v for k, v in config.PARAMS_BASE_RF.items() if k not in params}
    return lambda: RandomForestClassifier(**base, **params)


def crear_logistica(params: Dict[str, Any]) -> Callable[[], LogisticRegression]:
    base = {k: v for k, v in config.PARAMS_BASE_LOGREG.items() if k not in params}
    return lambda: LogisticRegression(**base, **params)


def buscar_hiperparametros(intentos: Dict[str, int] = None) -> Dict[str, Any]:
    """
    Optimiza el AUC promedio de la CV. No se optimiza el KS porque en folds de
    ~168 casos (unos 42 morosos) el KS por deciles salta mucho entre intentos.
    """
    intentos = intentos or config.INTENTOS_OPTUNA

    df_train = pd.read_csv(config.RUTA_TRAIN)
    preprocesador = Preprocesador.cargar(config.RUTA_PREPROCESADOR)
    X = preprocesador.transform(df_train)
    y = df_train[config.COLUMNA_OBJETIVO].values

    n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    scale_pos_weight = n_neg / n_pos
    print(f"Train: {X.shape} | {n_pos} morosos / {n_neg} cumplidores | scale_pos_weight = {scale_pos_weight:.2f}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    muestreador = lambda: optuna.samplers.TPESampler(seed=config.SEMILLA)

    def espacio_xgb(trial):
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

    def espacio_rf(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        }

    def espacio_lr(trial):
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}

    busquedas = {
        "XGBoost": (espacio_xgb, lambda p: crear_xgboost(p, scale_pos_weight)),
        "Random_Forest": (espacio_rf, crear_random_forest),
        "Regresion_Logistica": (espacio_lr, crear_logistica),
    }

    resultados = {}
    for nombre, (espacio, crear) in busquedas.items():
        # los params base con la misma CV, para comparar
        base = validacion_cruzada(crear({}), X, y)

        def objetivo(trial):
            metricas = validacion_cruzada(crear(espacio(trial)), X, y)
            trial.set_user_attr("cv_ks_pct", metricas["cv_ks_pct"])
            return metricas["cv_auc"]

        estudio = optuna.create_study(direction="maximize", sampler=muestreador())
        estudio.optimize(objetivo, n_trials=intentos[nombre])

        mejor = estudio.best_trial
        resultados[nombre] = {
            "mejores_params": mejor.params,
            "cv_auc": round(mejor.value, 4),
            "cv_ks_pct": mejor.user_attrs["cv_ks_pct"],
            "base_cv_auc": base["cv_auc"],
            "base_cv_ks_pct": base["cv_ks_pct"],
            "intentos": intentos[nombre],
        }
        print(f"{nombre:<20} | CV AUC base {base['cv_auc']:.4f} -> optimizado {mejor.value:.4f} "
              f"| CV KS base {base['cv_ks_pct']:.2f}% -> {mejor.user_attrs['cv_ks_pct']:.2f}% "
              f"| {intentos[nombre]} intentos")

    salida = {
        "metodo": "Optuna TPE + validación cruzada estratificada de 5 folds sobre train",
        "metrica_objetivo": "AUC promedio de validación cruzada",
        "random_state": config.SEMILLA,
        "modelos": resultados,
    }
    with open(config.RUTA_MEJORES_PARAMS, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"guardado en {config.RUTA_MEJORES_PARAMS}")
    return salida


def cargar_mejores_params(nombre_modelo: str) -> Dict[str, Any]:
    """Devuelve los mejores hiperparámetros de un modelo, o {} si no se ha corrido la búsqueda."""
    if not config.RUTA_MEJORES_PARAMS.exists():
        return {}
    with open(config.RUTA_MEJORES_PARAMS, "r", encoding="utf-8") as f:
        return json.load(f)["modelos"].get(nombre_modelo, {}).get("mejores_params", {})


if __name__ == "__main__":
    buscar_hiperparametros()
