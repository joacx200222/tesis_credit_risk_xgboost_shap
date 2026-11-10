"""Imputación, escalado mediana/IQR, One-Hot y partición train/val/test."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any

from src import config
from src.ingenieria_variables import cargar_encuesta


class Preprocesador:
    """
    Preprocesador hecho a mano (en vez de un ColumnTransformer) para poder
    guardarlo en JSON y conservar nombres de columna legibles para SHAP.
    """

    def __init__(
        self,
        numericas: List[str] = None,
        categoricas: List[str] = None
    ):
        self.numericas = list(numericas or config.VARIABLES_NUMERICAS)
        self.categoricas = list(categoricas or config.VARIABLES_CATEGORICAS)

        self.medianas_: Dict[str, float] = {}
        self.mediana_iqr_: Dict[str, Tuple[float, float]] = {}  # (mediana, IQR)
        self.modas_: Dict[str, str] = {}
        self.categorias_: Dict[str, List[Any]] = {}
        self.nombres_salida_: List[str] = []
        self.ajustado_: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "Preprocesador":
        for col in self.numericas:
            if col in X.columns:
                med = float(X[col].median())
                q25 = float(X[col].quantile(0.25))
                q75 = float(X[col].quantile(0.75))
                iqr = float(q75 - q25)
                if iqr <= 0:
                    iqr = 1.0
                self.medianas_[col] = med
                self.mediana_iqr_[col] = (med, iqr)
            else:
                self.medianas_[col] = 0.0
                self.mediana_iqr_[col] = (0.0, 1.0)

        self.nombres_salida_ = list(self.numericas)

        for col in self.categoricas:
            if col in X.columns:
                moda = str(X[col].mode().iloc[0]) if not X[col].dropna().empty else "Desconocido"
                categorias = sorted([str(c) for c in X[col].dropna().unique().tolist()])
                self.modas_[col] = moda
                self.categorias_[col] = categorias

                for cat in categorias:
                    self.nombres_salida_.append(f"{col}_{cat}")
            else:
                self.modas_[col] = "Desconocido"
                self.categorias_[col] = []

        self.ajustado_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.ajustado_:
            raise ValueError("El preprocesador debe ser ajustado (fit) antes de transformar datos.")

        X_copia = X.copy()

        bloques_num = []
        for col in self.numericas:
            if col in X_copia.columns:
                valores = X_copia[col].fillna(self.medianas_[col]).values.astype(float)
            else:
                valores = np.full(len(X_copia), self.medianas_[col], dtype=float)

            med, iqr = self.mediana_iqr_[col]
            escalado = (valores - med) / (iqr if iqr > 0 else 1.0)
            bloques_num.append(escalado.reshape(-1, 1))

        bloques_cat = []
        for col in self.categoricas:
            if col in X_copia.columns:
                valores = X_copia[col].fillna(self.modas_[col]).astype(str).values
            else:
                valores = np.full(len(X_copia), str(self.modas_[col]))

            categorias = self.categorias_[col]
            for cat in categorias:
                columna = (valores == str(cat)).astype(float).reshape(-1, 1)
                bloques_cat.append(columna)

        bloques = bloques_num + bloques_cat
        return np.hstack(bloques) if bloques else np.empty((len(X_copia), 0))

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        return self.nombres_salida_

    def guardar(self, ruta: Path = None):
        ruta_json = Path(ruta or config.RUTA_PREPROCESADOR).with_suffix(".json")
        ruta_json.parent.mkdir(parents=True, exist_ok=True)

        datos = {
            "numericas": self.numericas,
            "categoricas": self.categoricas,
            "medianas": self.medianas_,
            "mediana_iqr": {k: [float(v[0]), float(v[1])] for k, v in self.mediana_iqr_.items()},
            "modas": self.modas_,
            "categorias": self.categorias_,
            "nombres_salida": self.nombres_salida_,
            "ajustado": self.ajustado_
        }
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)

    @classmethod
    def cargar(cls, ruta: Path = None) -> "Preprocesador":
        ruta_json = Path(ruta or config.RUTA_PREPROCESADOR).with_suffix(".json")
        with open(ruta_json, "r", encoding="utf-8") as f:
            datos = json.load(f)

        instancia = cls(
            numericas=datos["numericas"],
            categoricas=datos["categoricas"]
        )
        instancia.medianas_ = {k: float(v) for k, v in datos["medianas"].items()}
        instancia.mediana_iqr_ = {k: (float(v[0]), float(v[1])) for k, v in datos["mediana_iqr"].items()}
        instancia.modas_ = datos["modas"]
        instancia.categorias_ = datos["categorias"]
        instancia.nombres_salida_ = datos["nombres_salida"]
        instancia.ajustado_ = datos["ajustado"]
        return instancia


def particion_estratificada(
    df: pd.DataFrame,
    columna_objetivo: str = config.COLUMNA_OBJETIVO,
    prop_train: float = config.PROP_TRAIN,
    prop_val: float = config.PROP_VAL,
    semilla: int = config.SEMILLA
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 estratificado por la variable objetivo."""
    np.random.seed(semilla)
    partes_train, partes_val, partes_test = [], [], []

    for _, grupo in df.groupby(columna_objetivo):
        mezclado = grupo.sample(frac=1.0, random_state=semilla)
        n = len(mezclado)
        n_train = int(n * prop_train)
        n_val = int(n * prop_val)

        partes_train.append(mezclado.iloc[:n_train])
        partes_val.append(mezclado.iloc[n_train:n_train + n_val])
        partes_test.append(mezclado.iloc[n_train + n_val:])

    df_train = pd.concat(partes_train).sample(frac=1.0, random_state=semilla).reset_index(drop=True)
    df_val = pd.concat(partes_val).sample(frac=1.0, random_state=semilla).reset_index(drop=True)
    df_test = pd.concat(partes_test).sample(frac=1.0, random_state=semilla).reset_index(drop=True)

    return df_train, df_val, df_test


def preparar_datos():
    config.DIR_PROCESADOS.mkdir(parents=True, exist_ok=True)
    config.DIR_CRUDOS.mkdir(parents=True, exist_ok=True)
    config.DIR_MODELOS.mkdir(parents=True, exist_ok=True)

    df_indices = cargar_encuesta(config.RUTA_DTA)

    df_indices.to_csv(config.RUTA_CSV_COMPLETO, index=False)

    df_train, df_val, df_test = particion_estratificada(df_indices)

    df_train.to_csv(config.RUTA_TRAIN, index=False)
    df_val.to_csv(config.RUTA_VAL, index=False)
    df_test.to_csv(config.RUTA_TEST, index=False)

    # el preprocesador se ajusta solo con train
    preprocesador = Preprocesador()
    preprocesador.fit(df_train)
    preprocesador.guardar(config.RUTA_PREPROCESADOR)

    for nombre, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"{nombre:<5} {len(d):>4} filas, mora {d[config.COLUMNA_OBJETIVO].mean()*100:.2f}%")
    print(f"{len(preprocesador.get_feature_names_out())} columnas después del One-Hot")


if __name__ == "__main__":
    preparar_datos()
