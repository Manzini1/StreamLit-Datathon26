from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_CANDIDATES = [
    "inde", "ida", "ieg", "iaa", "ips", "ipp", "ipv",
    "idade", "fase_ordem", "inde_prev",
]

CATEGORICAL_CANDIDATES = [
    "fase", "pedra", "genero", "instituicao_de_ensino", "ano_referencia"
]

LEAKAGE_COLUMNS = {
    "defasagem",
    "defasagem_next",
    "risco_defasagem_atual",
    "risco_defasagem_severo",
    "piora_defasagem_prox_ano",
    "ian",  # proxy muito próximo da defasagem; removido por cautela
}


@dataclass
class TrainConfig:
    target_col: str = "auto"  # auto | risco_defasagem_atual | risco_defasagem_severo | piora_defasagem_prox_ano
    threshold: float = 0.50
    random_state: int = 42
    min_labeled_rows: int = 50
    artifacts_dir: str | Path = "artifacts"


def choose_target(df: pd.DataFrame, requested: str = "auto", min_labeled_rows: int = 50) -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"Target '{requested}' não encontrado no dataframe.")
        return requested

    priority = ["piora_defasagem_prox_ano", "risco_defasagem_atual", "risco_defasagem_severo"]
    for t in priority:
        if t in df.columns and df[t].notna().sum() >= min_labeled_rows:
            # evita target quase constante
            vals = pd.to_numeric(df[t], errors="coerce").dropna().astype(int)
            if vals.nunique() >= 2:
                return t

    raise ValueError("Nenhum target utilizável encontrado (ou sem volume suficiente).")


def _sanitize_for_sklearn(X: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    """Normaliza dtypes/missings para evitar erros com pandas StringDtype/pd.NA no scikit-learn."""
    X = X.copy()

    # Numéricas: força float (np.nan como missing)
    for c in numeric_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")

    # Categóricas: converte para object e troca pd.NA por np.nan
    for c in categorical_cols:
        if c in X.columns:
            s = X[c]
            # evita manter StringDtype/BooleanDtype/Int64 nullable no sklearn
            s = s.astype("object")
            s = s.where(pd.notna(s), np.nan)
            X[c] = s

    return X


def build_feature_frame(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]]]:
    work = df.copy()

    # Garante tipos numéricos em candidatos numéricos
    for c in NUMERIC_CANDIDATES:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    # Target binário
    y = pd.to_numeric(work[target_col], errors="coerce")
    valid_y = y.notna()
    y = y.loc[valid_y].astype(int)
    work = work.loc[valid_y].copy()

    # Remove classes inválidas caso existam
    valid_class = y.isin([0, 1])
    y = y.loc[valid_class]
    work = work.loc[valid_class].copy()

    # Seleção de features disponíveis
    numeric_cols = [c for c in NUMERIC_CANDIDATES if c in work.columns and c not in LEAKAGE_COLUMNS and c != target_col]
    categorical_cols = [c for c in CATEGORICAL_CANDIDATES if c in work.columns and c not in LEAKAGE_COLUMNS and c != target_col]

    if not numeric_cols and not categorical_cols:
        raise ValueError("Nenhuma feature disponível para treinar o modelo.")

    X = work[numeric_cols + categorical_cols].copy()
    X = _sanitize_for_sklearn(X, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    schema = {"numeric": numeric_cols, "categorical": categorical_cols}
    return X, y, schema


def temporal_or_random_split(df_indexed: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """Split temporal usando ano mais recente no teste quando possível.

    Retorna (test_mask, test_year). Se o split temporal não gerar classes suficientes
    em treino e teste, cai para holdout aleatório estratificado.
    """
    y = pd.Series(y).reindex(df_indexed.index)

    def _is_valid(mask: pd.Series) -> bool:
        if mask is None:
            return False
        mask = pd.Series(mask, index=df_indexed.index).fillna(False).astype(bool)
        if mask.sum() < 20 or (~mask).sum() < 20:
            return False
        y_tr = y.loc[~mask].dropna().astype(int)
        y_te = y.loc[mask].dropna().astype(int)
        return y_tr.nunique() >= 2 and y_te.nunique() >= 2

    if "ano_referencia" in df_indexed.columns:
        years = pd.to_numeric(df_indexed["ano_referencia"], errors="coerce").dropna()
        uniq_years = sorted(years.unique())
        if len(uniq_years) >= 2:
            for test_year in reversed(uniq_years):
                test_mask = (pd.to_numeric(df_indexed["ano_referencia"], errors="coerce") == test_year).fillna(False)
                if _is_valid(test_mask):
                    return test_mask, test_year

    # fallback: holdout aleatório estratificado simples manual
    rng = np.random.RandomState(random_state)
    idx = df_indexed.index.to_numpy()
    y_vals = y.reindex(df_indexed.index)

    pos = idx[y_vals.values == 1]
    neg = idx[y_vals.values == 0]

    if len(pos) == 0 or len(neg) == 0:
        test_mask = df_indexed.index.to_series().isin([])
        return test_mask, None

    rng.shuffle(pos)
    rng.shuffle(neg)

    # Garante pelo menos 1 amostra por classe no treino e no teste, quando possível.
    n_pos_test = min(max(1, int(round(0.2 * len(pos)))), max(1, len(pos) - 1))
    n_neg_test = min(max(1, int(round(0.2 * len(neg)))), max(1, len(neg) - 1))

    test_idx = set(pos[:n_pos_test]).union(set(neg[:n_neg_test]))
    test_mask = df_indexed.index.to_series().isin(test_idx)
    return test_mask, None


def build_models(schema: dict[str, list[str]], random_state: int = 42) -> dict[str, Pipeline]:
    numeric_cols = schema["numeric"]
    categorical_cols = schema["categorical"]

    transformers = []
    if numeric_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    models = {
        "log_reg": Pipeline([
            ("preprocess", pre),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", pre),
            ("model", RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
    }
    return models


def evaluate_binary(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = pd.Series(y_true).astype(int)
    y_prob = pd.Series(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if y_true.nunique() > 1 else np.nan,
        "average_precision": float(average_precision_score(y_true, y_prob)) if y_true.nunique() > 1 else np.nan,
        "base_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }

    # threshold sugerido por F1 (para classe positiva)
    try:
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        f1 = (2 * prec * rec) / (prec + rec + 1e-12)
        if len(thr) > 0:
            best_i = int(np.nanargmax(f1[:-1]))
            metrics["threshold_sugerido_f1"] = float(thr[best_i])
            metrics["f1_max"] = float(f1[:-1][best_i])
    except Exception:
        pass
    return metrics


def _safe_feature_names(preprocess: ColumnTransformer) -> list[str]:
    try:
        return list(preprocess.get_feature_names_out())
    except Exception:
        names = []
        for _, _, cols in preprocess.transformers_:
            if isinstance(cols, list):
                names.extend(cols)
        return names


def get_feature_importance_df(fitted_pipe: Pipeline) -> pd.DataFrame:
    pre = fitted_pipe.named_steps["preprocess"]
    model = fitted_pipe.named_steps["model"]
    feat_names = _safe_feature_names(pre)

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feat_names, name="importance")
    elif hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        imp = pd.Series(np.abs(coef), index=feat_names, name="importance")
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    out = imp.sort_values(ascending=False).reset_index()
    out.columns = ["feature", "importance"]
    return out


def train_risk_model(df: pd.DataFrame, config: TrainConfig | None = None) -> dict[str, Any]:
    config = config or TrainConfig()

    def _train_for_target(target_col: str) -> dict[str, Any]:
        X, y, schema = build_feature_frame(df, target_col=target_col)

        if y.nunique() < 2:
            raise ValueError(f"Target '{target_col}' possui classe única após limpeza.")

        split_ref = X.copy()
        test_mask, test_year = temporal_or_random_split(split_ref, y, random_state=config.random_state)
        X_train, X_test = X.loc[~test_mask], X.loc[test_mask]
        y_train, y_test = y.loc[~test_mask], y.loc[test_mask]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            raise ValueError(
                f"Split inválido para target '{target_col}': classe única em treino ou teste."
            )

        models = build_models(schema, random_state=config.random_state)
        fitted = {}
        scores = []
        for name, pipe in models.items():
            pipe.fit(X_train, y_train)
            prob = pipe.predict_proba(X_test)[:, 1]
            metrics = evaluate_binary(y_test, prob, threshold=config.threshold)
            fitted[name] = {"pipeline": pipe, "metrics": metrics, "y_prob_test": prob}
            scores.append((name, metrics.get("average_precision", np.nan), metrics.get("roc_auc", np.nan)))

        scores_sorted = sorted(
            scores,
            key=lambda x: (np.nan_to_num(x[1], nan=-1), np.nan_to_num(x[2], nan=-1)),
            reverse=True,
        )
        best_name = scores_sorted[0][0]
        best_pipe = fitted[best_name]["pipeline"]

        all_prob = best_pipe.predict_proba(X)[:, 1]
        scored = X.copy()
        scored[target_col] = y.values
        scored["prob_risco"] = all_prob
        scored["pred_risco"] = (scored["prob_risco"] >= config.threshold).astype(int)

        fi_df = get_feature_importance_df(best_pipe)

        result = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "target_col": target_col,
            "threshold": config.threshold,
            "split": {
                "type": "temporal_latest_valid_year" if test_year is not None else "random_holdout",
                "test_year": int(test_year) if test_year is not None else None,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
            },
            "schema": schema,
            "candidate_metrics": {k: v["metrics"] for k, v in fitted.items()},
            "best_model_name": best_name,
            "best_metrics": fitted[best_name]["metrics"],
            "pipeline": best_pipe,
            "feature_importance": fi_df,
            "y_test": y_test,
            "y_prob_test": fitted[best_name]["y_prob_test"],
            "X_test": X_test,
            "scored_labeled": scored,
        }
        return result

    if config.target_col != "auto":
        return _train_for_target(config.target_col)

    tried_errors: dict[str, str] = {}
    priority = ["piora_defasagem_prox_ano", "risco_defasagem_atual", "risco_defasagem_severo"]
    for t in priority:
        if t not in df.columns:
            continue
        vals = pd.to_numeric(df[t], errors="coerce").dropna()
        if len(vals) < config.min_labeled_rows:
            continue
        vals_int = pd.to_numeric(vals, errors="coerce").dropna().astype(int)
        if vals_int.nunique() < 2:
            continue
        try:
            res = _train_for_target(t)
            res["auto_target_fallback_log"] = tried_errors
            return res
        except Exception as e:
            tried_errors[t] = str(e)

    raise ValueError(
        "Nenhum target com split válido em modo auto. "
        f"Tentativas: {tried_errors}. "
        "Selecione 'risco_defasagem_atual' no app ou adicione mais dados rotulados."
    )


def save_model_bundle(train_result: dict[str, Any], outpath: str | Path) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "created_at": train_result["created_at"],
        "target_col": train_result["target_col"],
        "threshold": train_result["threshold"],
        "split": train_result["split"],
        "schema": train_result["schema"],
        "best_model_name": train_result["best_model_name"],
        "best_metrics": train_result["best_metrics"],
        "candidate_metrics": train_result["candidate_metrics"],
        "pipeline": train_result["pipeline"],
    }
    joblib.dump(bundle, outpath)
    return outpath


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def prepare_inference_frame(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    schema = bundle["schema"]
    cols = schema.get("numeric", []) + schema.get("categorical", [])
    X = df.copy()
    # garante colunas ausentes
    for c in cols:
        if c not in X.columns:
            X[c] = np.nan

    X = X[cols].copy()
    X = _sanitize_for_sklearn(
        X,
        numeric_cols=schema.get("numeric", []),
        categorical_cols=schema.get("categorical", []),
    )
    return X


def score_dataframe(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    X = prepare_inference_frame(df, bundle)
    probs = bundle["pipeline"].predict_proba(X)[:, 1]
    out = df.copy()
    out["prob_risco"] = probs
    thr = float(bundle.get("threshold", 0.5))
    out["pred_risco"] = (out["prob_risco"] >= thr).astype(int)
    return out
