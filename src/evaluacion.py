"""Métricas (AUC, PR-AUC, KS, F1), pérdida esperada, score y decisión."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from src import config


def curva_roc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (auc, fpr, tpr, umbrales). Misma lógica que sklearn.metrics.roc_curve."""
    orden = np.argsort(y_prob, kind="mergesort")[::-1]
    y_ordenado = y_true[orden]
    prob_ordenada = y_prob[orden]

    cambios = np.where(np.diff(prob_ordenada))[0]
    idx_umbrales = np.r_[cambios, y_true.size - 1]

    tps = np.cumsum(y_ordenado)[idx_umbrales]
    fps = 1 + idx_umbrales - tps

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5, np.array([0, 1]), np.array([0, 1]), np.array([1, 0])

    tpr = tps / n_pos
    fpr = fps / n_neg

    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    umbrales = np.r_[prob_ordenada[0] + 1e-5, prob_ordenada[idx_umbrales]]

    # np.trapz se renombró a np.trapezoid en numpy 2
    trapecio = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc_roc = float(trapecio(tpr, fpr))
    return round(auc_roc, 4), fpr, tpr, umbrales


def curva_pr(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Devuelve (average precision, recall, precision)."""
    orden = np.argsort(y_prob, kind="mergesort")[::-1]
    y_ordenado = y_true[orden]

    tps = np.cumsum(y_ordenado == 1)
    fps = np.cumsum(y_ordenado == 0)

    precision = tps / (tps + fps)
    n_pos = np.sum(y_true == 1)
    recall = tps / (n_pos if n_pos > 0 else 1)

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]

    pr_auc = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
    return round(max(0.0, pr_auc), 4), recall, precision


def estadistico_ks(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, pd.DataFrame]:
    """
    KS por deciles: máxima distancia entre el % acumulado de malos y de buenos.
    Devuelve (ks en %, PD mínima del decil donde ocurre, tabla por decil).
    """
    df_ks = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})

    df_ks["decil"] = pd.qcut(df_ks["y_prob"], q=10, duplicates="drop")

    tabla = df_ks.groupby("decil", observed=False).agg(
        total=("y_true", "count"),
        malos=("y_true", lambda x: (x == 1).sum()),
        buenos=("y_true", lambda x: (x == 0).sum()),
        prob_min=("y_prob", "min"),
        prob_max=("y_prob", "max")
    ).reset_index()

    # del decil más riesgoso al menos riesgoso
    tabla = tabla.sort_values("prob_min", ascending=False).reset_index(drop=True)

    total_malos = tabla["malos"].sum()
    total_buenos = tabla["buenos"].sum()

    tabla["pct_malos"] = tabla["malos"] / (total_malos if total_malos > 0 else 1)
    tabla["pct_buenos"] = tabla["buenos"] / (total_buenos if total_buenos > 0 else 1)

    tabla["acum_malos"] = tabla["pct_malos"].cumsum()
    tabla["acum_buenos"] = tabla["pct_buenos"].cumsum()

    tabla["dif_ks"] = (tabla["acum_malos"] - tabla["acum_buenos"]).abs()

    idx_max = tabla["dif_ks"].idxmax()
    ks_max = float(tabla.loc[idx_max, "dif_ks"]) * 100.0
    corte = float(tabla.loc[idx_max, "prob_min"])

    return round(ks_max, 2), round(corte, 4), tabla


def calcular_metricas(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    umbral: float = 0.50
) -> Dict[str, Any]:
    """Todas las métricas que van al benchmark; las de clasificación usan un umbral fijo."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= umbral).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)
    exactitud = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    auc_roc, _, _, _ = curva_roc(y_true, y_prob)
    pr_auc, _, _ = curva_pr(y_true, y_prob)
    ks, ks_corte, _ = estadistico_ks(y_true, y_prob)

    return {
        "umbral": round(umbral, 2),
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "ks_pct": ks,
        "ks_corte": ks_corte,
        "f1": round(f1, 4),
        "recall_morosos": round(recall, 4),
        "precision_morosos": round(precision, 4),
        "especificidad": round(especificidad, 4),
        "exactitud": round(exactitud, 4),
        "matriz_confusion": {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn
        }
    }


def perdida_esperada(
    pd_valor: float,
    monto: float,
    lgd: float = config.LGD
) -> float:
    """EL = PD * EAD * LGD, en soles. EAD es el monto solicitado."""
    return round(float(pd_valor) * float(monto) * float(lgd), 2)


def pd_a_score(pd_valor: float) -> int:
    """Score = 1000 * (1 - PD), recortado a [0, 1000]."""
    score = int(np.round(config.SCORE_MAX * (1.0 - pd_valor)))
    return int(np.clip(score, config.SCORE_MIN, config.SCORE_MAX))


def decision_crediticia(pd_valor: float) -> Tuple[str, str, str]:
    """Devuelve (decisión, color para la app, descripción de la política)."""
    if pd_valor <= config.UMBRAL_APROBACION:
        return "APROBADO", "#28a745", "Bajo Riesgo: Aprobación automática conforme a políticas de crédito."
    elif pd_valor <= config.UMBRAL_REVISION:
        return "REVISIÓN MANUAL", "#ffc107", "Riesgo Moderado: Requiere validación de aval o visita de campo por analista."
    else:
        return "RECHAZADO", "#dc3545", "Alto Riesgo: Excede el límite de tolerancia de cartera pesada."
