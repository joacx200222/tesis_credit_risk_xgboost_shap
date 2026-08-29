"""
==============================================================================
PROYECTO DE TESIS - UNIVERSIDAD DE LIMA
Carrera: Ingeniería de Sistemas
Título: Modelo de evaluación de riesgo crediticio para microfinanzas e
        inclusión financiera mediante XGBoost e Inteligencia Artificial
        Explicable (SHAP)
Archivo: src/data_preprocessing.py
Descripción: Pipeline de preprocesamiento de datos (CRISP-DM Fase 3),
             imputación estratégica, escalado robusto, codificación One-Hot y
             partición estratificada (Train / Val / Test).
==============================================================================
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any
import sys

# Importar configuración e ingeniería de características
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.feature_engineering import add_engineered_features


class CreditDataPreprocessor:
    """
    Transformador de datos para el modelo de scoring crediticio.
    Realiza imputación, escalado de variables numéricas y codificación One-Hot
    de variables categóricas preservando los nombres de características
    para compatibilidad directa con SHAP (TreeSHAP).
    """

    def __init__(
        self,
        numerical_features: List[str] = None,
        categorical_features: List[str] = None
    ):
        self.numerical_features = list(numerical_features or config.ALL_NUMERICAL_FEATURES)
        self.categorical_features = list(categorical_features or config.CATEGORICAL_FEATURES)
        
        # Parámetros aprendidos en el ajuste (Fit)
        self.num_medians_: Dict[str, float] = {}
        self.num_iqr_: Dict[str, Tuple[float, float]] = {}  # (median, IQR) para RobustScaler
        self.cat_modes_: Dict[str, str] = {}
        self.cat_categories_: Dict[str, List[Any]] = {}
        self.feature_names_out_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CreditDataPreprocessor":
        """
        Ajusta los parámetros estadísticos del preprocesador sobre el conjunto de entrenamiento.
        """
        # 1. Ajuste numérico (Mediana y Rango Intercuartílico IQR)
        for col in self.numerical_features:
            if col in X.columns:
                med = float(X[col].median())
                q25 = float(X[col].quantile(0.25))
                q75 = float(X[col].quantile(0.75))
                iqr = float(q75 - q25)
                if iqr == 0:
                    iqr = 1.0
                self.num_medians_[col] = med
                self.num_iqr_[col] = (med, iqr)
            else:
                self.num_medians_[col] = 0.0
                self.num_iqr_[col] = (0.0, 1.0)

        # 2. Ajuste categórico (Categorías únicas y moda)
        self.feature_names_out_ = list(self.numerical_features)
        
        for col in self.categorical_features:
            if col in X.columns:
                mode_val = str(X[col].mode().iloc[0]) if not X[col].dropna().empty else "Desconocido"
                cats = sorted([str(c) for c in X[col].dropna().unique().tolist()])
                self.cat_modes_[col] = mode_val
                self.cat_categories_[col] = cats
                
                # Nombres de variables codificadas One-Hot
                for cat in cats:
                    self.feature_names_out_.append(f"{col}_{cat}")
            else:
                self.cat_modes_[col] = "Desconocido"
                self.cat_categories_[col] = []

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforma los datos de entrada aplicando escalado robusto y One-Hot Encoding.
        """
        if not self.is_fitted_:
            raise ValueError("El preprocesador debe ser ajustado (fit) antes de transformar datos.")

        X_df = X.copy()
        
        # 1. Transformación numérica (Imputación por mediana + Robust Scaling)
        num_arrays = []
        for col in self.numerical_features:
            if col in X_df.columns:
                vals = X_df[col].fillna(self.num_medians_[col]).values.astype(float)
            else:
                vals = np.full(len(X_df), self.num_medians_[col], dtype=float)
            
            med, iqr = self.num_iqr_[col]
            scaled = (vals - med) / (iqr if iqr > 0 else 1.0)
            num_arrays.append(scaled.reshape(-1, 1))

        # 2. Transformación categórica (One-Hot Encoding)
        cat_arrays = []
        for col in self.categorical_features:
            if col in X_df.columns:
                vals = X_df[col].fillna(self.cat_modes_[col]).astype(str).values
            else:
                vals = np.full(len(X_df), str(self.cat_modes_[col]))

            cats = self.cat_categories_[col]
            for cat in cats:
                cat_encoded = (vals == str(cat)).astype(float).reshape(-1, 1)
                cat_arrays.append(cat_encoded)

        # Concatenación final
        all_parts = num_arrays + cat_arrays
        X_out = np.hstack(all_parts) if all_parts else np.empty((len(X_df), 0))
        return X_out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """Ajusta y transforma en un único paso."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Retorna los nombres de todas las columnas transformadas (útil para SHAP)."""
        return self.feature_names_out_

    def save(self, filepath: Path = None):
        """Serializa el preprocesador en disco en formato JSON altamente portable."""
        path = filepath or config.PREPROCESSOR_PATH
        path = Path(path).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "num_medians_": self.num_medians_,
            "num_iqr_": {k: [float(v[0]), float(v[1])] for k, v in self.num_iqr_.items()},
            "cat_modes_": self.cat_modes_,
            "cat_categories_": self.cat_categories_,
            "feature_names_out_": self.feature_names_out_,
            "is_fitted_": self.is_fitted_
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: Path = None) -> "CreditDataPreprocessor":
        """Carga el preprocesador serializado desde archivo JSON."""
        path = filepath or config.PREPROCESSOR_PATH
        path = Path(path).with_suffix(".json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        instance = cls(
            numerical_features=data["numerical_features"],
            categorical_features=data["categorical_features"]
        )
        instance.num_medians_ = {k: float(v) for k, v in data["num_medians_"].items()}
        instance.num_iqr_ = {k: (float(v[0]), float(v[1])) for k, v in data["num_iqr_"].items()}
        instance.cat_modes_ = data["cat_modes_"]
        instance.cat_categories_ = data["cat_categories_"]
        instance.feature_names_out_ = data["feature_names_out_"]
        instance.is_fitted_ = data["is_fitted_"]
        return instance


def stratified_split_dataset(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COLUMN,
    train_size: float = config.TRAIN_SIZE,
    val_size: float = config.VAL_SIZE,
    random_state: int = config.RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partición estratificada del conjunto de datos.
    """
    np.random.seed(random_state)
    train_dfs, val_dfs, test_dfs = [], [], []

    for _, group in df.groupby(target_col):
        shuffled = group.sample(frac=1.0, random_state=random_state)
        n = len(shuffled)
        n_train = int(n * train_size)
        n_val = int(n * val_size)

        train_dfs.append(shuffled.iloc[:n_train])
        val_dfs.append(shuffled.iloc[n_train:n_train + n_val])
        test_dfs.append(shuffled.iloc[n_train + n_val:])

    df_train = pd.concat(train_dfs).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df_val = pd.concat(val_dfs).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df_test = pd.concat(test_dfs).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return df_train, df_val, df_test


def prepare_and_save_data():
    """Ejecuta el pipeline completo de preparación de datos."""
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("--- [Fase 3 CRISP-DM: Preparación de Datos] ---")
    df_raw = pd.read_csv(config.RAW_DATASET_PATH)
    df_enriched = add_engineered_features(df_raw)

    df_train, df_val, df_test = stratified_split_dataset(df_enriched)

    # Persistencia de datasets
    df_train.to_csv(config.PROCESSED_TRAIN_PATH, index=False)
    df_val.to_csv(config.PROCESSED_VAL_PATH, index=False)
    df_test.to_csv(config.PROCESSED_TEST_PATH, index=False)

    # Ajustar y guardar preprocesador
    preprocessor = CreditDataPreprocessor()
    preprocessor.fit(df_train)
    preprocessor.save(config.PREPROCESSOR_PATH)

    print(f"[OK] Partición estratificada completada:")
    print(f"     Train : {len(df_train):,} registros ({df_train[config.TARGET_COLUMN].mean()*100:.2f}% default)")
    print(f"     Val   : {len(df_val):,} registros ({df_val[config.TARGET_COLUMN].mean()*100:.2f}% default)")
    print(f"     Test  : {len(df_test):,} registros ({df_test[config.TARGET_COLUMN].mean()*100:.2f}% default)")
    print(f"[OK] Preprocesador guardado en: {config.PREPROCESSOR_PATH.with_suffix('.json')}")
    print(f"     Total características codificadas: {len(preprocessor.get_feature_names_out())}")


if __name__ == "__main__":
    prepare_and_save_data()
