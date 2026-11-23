"""NR-IQA metrics: SRCC, KRCC, PLCC (raw and after 4-parameter logistic mapping), RMSE."""
import warnings
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import expit


def _logistic4(x, b1, b2, b3, b4):
    return (b1 - b2) * expit((x - b3) / np.abs(b4)) + b2


def logistic_map(pred: np.ndarray, target: np.ndarray):
    """Fit the monotonic logistic that maps predictions to the target scale. Returns (mapped, ok)."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    p0 = [target.max(), target.min(), pred.mean(), max(pred.std(), 1e-6) / 4]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(_logistic4, pred, target, p0=p0, maxfev=20000)
        mapped = _logistic4(pred, *popt)
        if not np.all(np.isfinite(mapped)):
            raise RuntimeError("non-finite fit")
        return mapped, True
    except (RuntimeError, ValueError):
        return pred, False


def _clean(v) -> float:
    v = float(v)
    return v if np.isfinite(v) else 0.0


def compute_metrics(pred, target) -> dict:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    assert pred.shape == target.shape and pred.size > 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant-input warnings from scipy
        srcc = _clean(stats.spearmanr(pred, target)[0])
        krcc = _clean(stats.kendalltau(pred, target)[0])
        plcc_raw = _clean(stats.pearsonr(pred, target)[0])
        mapped, ok = logistic_map(pred, target)
        plcc = _clean(stats.pearsonr(mapped, target)[0])
    rmse = float(np.sqrt(np.mean((mapped - target) ** 2)))
    return {"srcc": srcc, "plcc": plcc, "krcc": krcc, "plcc_raw": plcc_raw,
            "rmse": rmse, "n": int(pred.size), "logistic_fit_ok": bool(ok)}
