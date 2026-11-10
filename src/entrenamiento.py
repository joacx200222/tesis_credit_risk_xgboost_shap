import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from src import config
from src.preprocesamiento import Preprocesador
from src.evaluacion import calcular_metricas
from src.busqueda_hiperparametros import cargar_mejores_params


class ModeloXGBoost:
    """
    XGBClassifier con scale_pos_weight calculado del train (negativos / positivos),
    guardado en dos JSON: metadatos y modelo nativo.
    """
    def __init__(self, params: Dict[str, Any] = None):
        self.params = dict(params or config.PARAMS_BASE_XGBOOST)
        self.xgb: xgb.XGBClassifier = None
        self.nombres_variables: List[str] = []
        self.entrenado: bool = False
        self.importancias_: np.ndarray = None
        self.scale_pos_weight: float = float(config.SCALE_POS_WEIGHT)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, nombres_variables: list = None):
        self.nombres_variables = nombres_variables or [f"f_{i}" for i in range(X_train.shape[1])]

        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        self.scale_pos_weight = float(n_neg / (n_pos if n_pos > 0 else 1))

        params_fit = dict(self.params)
        params_fit["scale_pos_weight"] = self.scale_pos_weight

        self.xgb = xgb.XGBClassifier(**params_fit)

        if X_val is not None and y_val is not None:
            self.xgb.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.xgb.fit(X_train, y_train, verbose=False)

        self.importancias_ = self.xgb.feature_importances_
        self.entrenado = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.entrenado or self.xgb is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir.")
        return self.xgb.predict_proba(X)

    def predict(self, X: np.ndarray, umbral: float = 0.50) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return (prob >= umbral).astype(int)

    def guardar(self, ruta: Path = None):
        ruta_json = Path(ruta or config.RUTA_MODELO).with_suffix(".json")
        ruta_json.parent.mkdir(parents=True, exist_ok=True)

        datos = {
            "params": {k: v for k, v in self.params.items() if isinstance(v, (int, float, str, bool))},
            "scale_pos_weight": float(self.scale_pos_weight),
            "nombres_variables": self.nombres_variables,
            "importancias": self.importancias_.tolist() if self.importancias_ is not None else [],
            "entrenado": self.entrenado
        }
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)

        ruta_nativo = ruta_json.with_name(ruta_json.stem + "_nativo.json")
        self.xgb.save_model(str(ruta_nativo))

    @classmethod
    def cargar(cls, ruta: Path = None) -> "ModeloXGBoost":
        ruta_json = Path(ruta or config.RUTA_MODELO).with_suffix(".json")
        with open(ruta_json, "r", encoding="utf-8") as f:
            datos = json.load(f)

        instancia = cls(params=datos["params"])
        instancia.scale_pos_weight = datos["scale_pos_weight"]
        instancia.nombres_variables = datos["nombres_variables"]
        instancia.importancias_ = np.array(datos["importancias"], dtype=float) if datos["importancias"] else None
        instancia.entrenado = datos["entrenado"]

        ruta_nativo = ruta_json.with_name(ruta_json.stem + "_nativo.json")
        if not ruta_nativo.exists():
            raise FileNotFoundError(f"No se encontró el modelo nativo de XGBoost en: {ruta_nativo}")
        instancia.xgb = xgb.XGBClassifier()
        instancia.xgb.load_model(str(ruta_nativo))

        return instancia


def entrenar_y_comparar() -> Dict[str, Any]:
    """Entrena regresión logística, random forest y XGBoost (base y optimizados) y los compara en test."""

    df_train = pd.read_csv(config.RUTA_TRAIN)
    df_val = pd.read_csv(config.RUTA_VAL)
    df_test = pd.read_csv(config.RUTA_TEST)

    preprocesador = Preprocesador.cargar(config.RUTA_PREPROCESADOR)

    X_train = preprocesador.transform(df_train)
    y_train = df_train[config.COLUMNA_OBJETIVO].values

    X_val = preprocesador.transform(df_val)
    y_val = df_val[config.COLUMNA_OBJETIVO].values

    X_test = preprocesador.transform(df_test)
    y_test = df_test[config.COLUMNA_OBJETIVO].values

    nombres_variables = preprocesador.get_feature_names_out()

    print(f"train {X_train.shape}, mora {np.mean(y_train)*100:.2f}% | test {X_test.shape}, mora {np.mean(y_test)*100:.2f}%")

    # {} si todavía no se corrió busqueda_hiperparametros
    mejores_lr = cargar_mejores_params("Regresion_Logistica")
    mejores_rf = cargar_mejores_params("Random_Forest")
    mejores_xgb = cargar_mejores_params("XGBoost")
    optimizado = bool(mejores_lr and mejores_rf and mejores_xgb)
    print(f"Hiperparámetros: {'optimizados con Optuna' if optimizado else 'base (sin optimizar)'}")

    def evaluar(modelo):
        return calcular_metricas(y_test, modelo.predict_proba(X_test)[:, 1], umbral=0.50)

    lr_base = LogisticRegression(**config.PARAMS_BASE_LOGREG).fit(X_train, y_train)
    lr = LogisticRegression(**{**config.PARAMS_BASE_LOGREG, **mejores_lr}).fit(X_train, y_train)
    m_lr_base, m_lr = evaluar(lr_base), evaluar(lr)

    rf_base = RandomForestClassifier(**config.PARAMS_BASE_RF).fit(X_train, y_train)
    rf = RandomForestClassifier(**{**config.PARAMS_BASE_RF, **mejores_rf}).fit(X_train, y_train)
    m_rf_base, m_rf = evaluar(rf_base), evaluar(rf)

    xgb_base = ModeloXGBoost()
    xgb_base.fit(X_train, y_train, X_val=X_val, y_val=y_val, nombres_variables=nombres_variables)
    xgb_opt = ModeloXGBoost(params={**config.PARAMS_BASE_XGBOOST, **mejores_xgb})
    xgb_opt.fit(X_train, y_train, X_val=X_val, y_val=y_val, nombres_variables=nombres_variables)
    m_xgb_base, m_xgb = evaluar(xgb_base), evaluar(xgb_opt)

    xgb_opt.guardar(config.RUTA_MODELO)

    benchmark = {
        "hiperparametros_optimizados": optimizado,
        "comparacion_base_vs_optimizado": [
            {
                "Modelo": nombre,
                "KS base (%)": base["ks_pct"],
                "KS optimizado (%)": opt["ks_pct"],
                "AUC base": base["auc_roc"],
                "AUC optimizado": opt["auc_roc"],
            }
            for nombre, base, opt in [
                ("Regresión Logística", m_lr_base, m_lr),
                ("Random Forest", m_rf_base, m_rf),
                ("XGBoost", m_xgb_base, m_xgb),
            ]
        ],
        "modelos": {
            "Regresion_Logistica": {
                "nombre_mostrar": "Regresión Logística (Scorecard Tradicional)",
                "tipo": "Paramétrico Lineal",
                "metricas": m_lr
            },
            "Random_Forest": {
                "nombre_mostrar": "Random Forest Classifier",
                "tipo": "Ensamble Bagging",
                "metricas": m_rf
            },
            "XGBoost_Propuesto": {
                "nombre_mostrar": "XGBoost + Cost-Sensitive (Modelo de Tesis)",
                "tipo": "Gradient Boosting Decision Trees (GBDT)",
                "metricas": m_xgb
            }
        },
        "resumen_comparativo": [
            {
                "Modelo": "Regresión Logística",
                "AUC-ROC": m_lr["auc_roc"],
                "PR-AUC": m_lr["pr_auc"],
                "KS (%)": m_lr["ks_pct"],
                "F1-Score": m_lr["f1"],
                "Recall Morosos": m_lr["recall_morosos"]
            },
            {
                "Modelo": "Random Forest",
                "AUC-ROC": m_rf["auc_roc"],
                "PR-AUC": m_rf["pr_auc"],
                "KS (%)": m_rf["ks_pct"],
                "F1-Score": m_rf["f1"],
                "Recall Morosos": m_rf["recall_morosos"]
            },
            {
                "Modelo": "XGBoost (propuesto)",
                "AUC-ROC": m_xgb["auc_roc"],
                "PR-AUC": m_xgb["pr_auc"],
                "KS (%)": m_xgb["ks_pct"],
                "F1-Score": m_xgb["f1"],
                "Recall Morosos": m_xgb["recall_morosos"]
            }
        ]
    }

    with open(config.RUTA_BENCHMARK, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)

    print()
    print(f"{'Modelo':<30} | {'AUC-ROC':<9} | {'PR-AUC':<8} | {'KS (%)':<8} | {'Recall':<8}")
    for r in benchmark["resumen_comparativo"]:
        print(f"{r['Modelo']:<30} | {r['AUC-ROC']:<9.4f} | {r['PR-AUC']:<8.4f} | {r['KS (%)']:<8.2f} | {r['Recall Morosos']:<8.4f}")
    print()
    print(f"{'Base vs optimizado (test)':<30} | {'KS base':<8} | {'KS opt':<8} | {'AUC base':<8} | {'AUC opt':<8}")
    for r in benchmark["comparacion_base_vs_optimizado"]:
        print(f"{r['Modelo']:<30} | {r['KS base (%)']:<8.2f} | {r['KS optimizado (%)']:<8.2f} | {r['AUC base']:<8.4f} | {r['AUC optimizado']:<8.4f}")

    return benchmark


if __name__ == "__main__":
    entrenar_y_comparar()
