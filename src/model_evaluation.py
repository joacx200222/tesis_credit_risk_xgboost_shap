"""Métricas (AUC, PR-AUC, KS, F1), pérdida esperada, score y decisión."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from src import config


def compute_roc_curve_and_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (auc, fpr, tpr, umbrales). Misma lógica que sklearn.metrics.roc_curve."""
    desc_score_indices = np.argsort(y_prob, kind="mergesort")[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_prob_sorted = y_prob[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_prob_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5, np.array([0, 1]), np.array([0, 1]), np.array([1, 0])

    tpr = tps / n_pos
    fpr = fps / n_neg

    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    thresholds = np.r_[y_prob_sorted[0] + 1e-5, y_prob_sorted[threshold_idxs]]

    # np.trapz se renombró a np.trapezoid en numpy 2
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc_roc = float(trapz_func(tpr, fpr))
    return round(auc_roc, 4), fpr, tpr, thresholds


def compute_pr_curve_and_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Devuelve (average precision, recall, precision)."""
    desc_score_indices = np.argsort(y_prob, kind="mergesort")[::-1]
    y_true_sorted = y_true[desc_score_indices]

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    precision = tps / (tps + fps)
    n_pos = np.sum(y_true == 1)
    recall = tps / (n_pos if n_pos > 0 else 1)

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]

    pr_auc = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
    return round(max(0.0, pr_auc), 4), recall, precision


def compute_kolmogorov_smirnov(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, pd.DataFrame]:
    """
    KS por deciles: máxima distancia entre el % acumulado de malos y de buenos.
    Devuelve (ks en %, PD mínima del decil donde ocurre, tabla por decil).
    """
    df_ks = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})

    df_ks["decil"] = pd.qcut(df_ks["y_prob"], q=10, duplicates="drop")

    grouped = df_ks.groupby("decil", observed=False).agg(
        total=("y_true", "count"),
        malos=("y_true", lambda x: (x == 1).sum()),
        buenos=("y_true", lambda x: (x == 0).sum()),
        min_prob=("y_prob", "min"),
        max_prob=("y_prob", "max")
    ).reset_index()

    # del decil más riesgoso al menos riesgoso
    grouped = grouped.sort_values("min_prob", ascending=False).reset_index(drop=True)

    total_malos = grouped["malos"].sum()
    total_buenos = grouped["buenos"].sum()

    grouped["pct_malos"] = grouped["malos"] / (total_malos if total_malos > 0 else 1)
    grouped["pct_buenos"] = grouped["buenos"] / (total_buenos if total_buenos > 0 else 1)

    grouped["cum_malos"] = grouped["pct_malos"].cumsum()
    grouped["cum_buenos"] = grouped["pct_buenos"].cumsum()

    grouped["ks_diff"] = (grouped["cum_malos"] - grouped["cum_buenos"]).abs()

    max_ks_idx = grouped["ks_diff"].idxmax()
    max_ks = float(grouped.loc[max_ks_idx, "ks_diff"]) * 100.0
    optimal_cutoff = float(grouped.loc[max_ks_idx, "min_prob"])

    return round(max_ks, 2), round(optimal_cutoff, 4), grouped


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """Todas las métricas que van al benchmark; las de clasificación usan un umbral fijo."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    auc_roc, _, _, _ = compute_roc_curve_and_auc(y_true, y_prob)
    pr_auc, _, _ = compute_pr_curve_and_auc(y_true, y_prob)
    ks_stat, ks_cutoff, _ = compute_kolmogorov_smirnov(y_true, y_prob)

    return {
        "threshold": round(threshold, 2),
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "ks_stat_pct": ks_stat,
        "ks_optimal_cutoff": ks_cutoff,
        "f1_score": round(f1_score, 4),
        "recall_morosos": round(recall, 4),
        "precision_morosos": round(precision, 4),
        "specificity": round(specificity, 4),
        "accuracy": round(accuracy, 4),
        "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn
        }
    }


def calculate_expected_loss(
    pd_value: float,
    ead_monto: float,
    lgd: float = config.DEFAULT_LGD
) -> float:
    """EL = PD * EAD * LGD, en soles. EAD es el monto solicitado."""
    return round(float(pd_value) * float(ead_monto) * float(lgd), 2)


def convert_pd_to_credit_score(pd_value: float) -> int:
    """Score = 1000 * (1 - PD), recortado a [0, 1000]."""
    score = int(np.round(config.SCORE_MAX * (1.0 - pd_value)))
    return int(np.clip(score, config.SCORE_MIN, config.SCORE_MAX))


def get_credit_decision(pd_value: float) -> Tuple[str, str, str]:
    """Devuelve (decisión, color para la app, descripción de la política)."""
    if pd_value <= config.THRESHOLD_LOW_RISK:
        return "APROBADO", "#28a745", "Bajo Riesgo: Aprobación automática conforme a políticas de crédito."
    elif pd_value <= config.THRESHOLD_MEDIUM_RISK:
        return "REVISIÓN MANUAL", "#ffc107", "Riesgo Moderado: Requiere validación de aval o visita de campo por analista."
    else:
        return "RECHAZADO", "#dc3545", "Alto Riesgo: Excede el límite de tolerancia de cartera pesada."
