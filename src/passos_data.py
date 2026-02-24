from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_SHEET_ID = "1td91KoeSgXrUrCVOUkLmONG9Go3LVcXpcNEw_XrL2R0"
DEFAULT_SHEETS = ("PEDE2022", "PEDE2023", "PEDE2024")


@dataclass
class LoadConfig:
    sheet_id: str = DEFAULT_SHEET_ID
    sheet_names: tuple[str, ...] = DEFAULT_SHEETS
    cache_dir: str | Path = "data_raw"
    force_refresh: bool = False


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(ch)
    )


def slugify_col(col: str) -> str:
    col = _strip_accents(str(col)).strip().lower()
    col = col.replace("%", " perc ").replace("º", " ")
    col = re.sub(r"[\n\r\t]+", " ", col)
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def _as_numeric(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "biufc":
        return pd.to_numeric(series, errors="coerce")

    # Usa StringDtype para evitar perder o accessor .str quando a coluna
    # vira toda nula após limpeza (pandas pode mudar para float/object).
    s = series.astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    if s.isna().all():
        return pd.Series(np.nan, index=series.index, dtype="float64")

    # Garante dtype string antes das próximas operações .str
    s = s.astype("string")

    # Remove textos óbvios
    # Ex.: "Avaliador-11"
    # Mantém negativos e decimais com vírgula ou ponto.
    s = s.str.replace(r"\s+", "", regex=True)
    # Se parecer data, não converte.
    looks_like_date = s.str.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$", na=False)
    if looks_like_date.mean() > 0.4:
        return pd.to_numeric(pd.Series([np.nan] * len(s), index=s.index), errors="coerce")

    # Remove separador de milhar (.) quando vírgula é decimal
    s = s.str.replace(r"(?<=\d)\.(?=\d{3}(\D|$))", "", regex=True)
    s = s.str.replace(",", ".", regex=False)
    # Mantém apenas formatos numéricos
    mask_num = s.str.match(r"^-?\d+(\.\d+)?$", na=False)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out.loc[mask_num] = pd.to_numeric(s.loc[mask_num], errors="coerce")
    return out


def _detect_header_row(df_raw: pd.DataFrame, max_scan: int = 10) -> int:
    # Procura linha com várias colunas-chave frequentes no dataset
    tokens = {"ra", "inde", "fase", "turma", "idade", "genero", "ian", "ida", "ieg", "ipv"}
    best_idx, best_score = 0, -1
    scan_limit = min(max_scan, len(df_raw))
    for i in range(scan_limit):
        vals = [slugify_col(v) for v in df_raw.iloc[i].tolist()]
        score = sum(any(t in v for t in tokens) for v in vals)
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


def _clean_headers(headers: Iterable[str]) -> list[str]:
    cols = []
    seen = {}
    for c in headers:
        name = slugify_col(c)
        if not name:
            name = "coluna"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        cols.append(name)
    return cols


def read_sheet_csv_public(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    """
    Lê uma aba pública do Google Sheets via endpoint gviz (CSV).
    Funciona para planilhas públicas/compartilhadas por link.
    """
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    # Tenta sniffing de separador primeiro (há casos com ; e vírgula decimal)
    df = pd.read_csv(url, dtype=str, sep=None, engine="python")
    # Se a primeira linha virou header ruim, tenta fallback sem header
    if df.shape[1] <= 2 and any("Unnamed" in str(c) for c in df.columns):
        df = pd.read_csv(url, dtype=str, header=None)
    return df


def save_cache(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_cached_or_remote(sheet_id: str, sheet_name: str, cache_dir: str | Path, force_refresh: bool = False) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{sheet_name}.csv"
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path, dtype=str)
    df = read_sheet_csv_public(sheet_id, sheet_name)
    save_cache(df, cache_path)
    return df


def standardize_pede_sheet(df_raw: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    df_raw = df_raw.copy()

    # Se vier sem headers adequados, detecta linha de cabeçalho
    if all(str(c).startswith("Unnamed") for c in df_raw.columns):
        hdr_idx = _detect_header_row(df_raw)
        header = df_raw.iloc[hdr_idx].tolist()
        df = df_raw.iloc[hdr_idx + 1:].copy()
        df.columns = _clean_headers(header)
    else:
        df = df_raw.copy()
        df.columns = _clean_headers(df.columns)

    # Remove linhas totalmente vazias
    df = df.dropna(how="all").reset_index(drop=True)

    # Remove linhas que repetem cabeçalho no meio
    if "ra" in df.columns:
        df = df[df["ra"].astype(str).str.lower().ne("ra")]

    # Ano da aba
    year_match = re.search(r"(20\d{2})", sheet_name)
    df["ano_referencia"] = int(year_match.group(1)) if year_match else np.nan
    df["sheet_name"] = sheet_name

    # Tenta converter datas
    for c in list(df.columns):
        if any(k in c for k in ["data", "nasc"]):
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=False)
            except Exception:
                pass

    # Converte colunas numéricas (heurística)
    maybe_numeric_keywords = [
        "inde", "ida", "ieg", "iaa", "ips", "ipp", "ipv", "ian", "idade",
        "mat", "por", "ing", "cg", "cf", "ct", "defasagem", "n_av", "n_avali"
    ]
    for c in df.columns:
        if c in {"ra"}:
            continue
        if any(k in c for k in maybe_numeric_keywords):
            num = _as_numeric(df[c])
            # só substitui se gerou informação
            if num.notna().sum() >= max(3, int(0.2 * len(df))):
                df[c] = num

    # Normalização de categorias importantes
    for c in ["fase", "pedra_2023", "pedra_23", "pedra_22", "pedra_21", "pedra_20", "pedra", "genero", "instituicao_de_ensino"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip().replace({"": pd.NA})

    # Alias de colunas para uso mais fácil
    aliases = {}
    for c in df.columns:
        if c == "inde_2023":
            aliases.setdefault("inde_prev", c)
        if c == "inde_23":
            aliases.setdefault("inde_prev", c)
        if c == "inde":
            aliases.setdefault("inde", c)
        if c == "ida":
            aliases.setdefault("ida", c)
        if c == "ieg":
            aliases.setdefault("ieg", c)
        if c == "iaa":
            aliases.setdefault("iaa", c)
        if c == "ips":
            aliases.setdefault("ips", c)
        if c == "ipp":
            aliases.setdefault("ipp", c)
        if c == "ipv":
            aliases.setdefault("ipv", c)
        if c == "ian":
            aliases.setdefault("ian", c)
        if c == "defasagem":
            aliases.setdefault("defasagem", c)
        if c == "fase_ideal":
            aliases.setdefault("fase_ideal", c)
        if c == "ra":
            aliases.setdefault("ra", c)
        if c == "pedra_2023":
            aliases.setdefault("pedra", c)
        if c == "pedra_23":
            aliases.setdefault("pedra", c)
        if c == "pedra_22":
            aliases.setdefault("pedra", c)
    for alias, col in aliases.items():
        if alias not in df.columns:
            df[alias] = df[col]

    # Limpa colunas de string vazias
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    return df.reset_index(drop=True)


def load_all_years(config: LoadConfig | None = None) -> pd.DataFrame:
    config = config or LoadConfig()
    frames = []
    for sheet in config.sheet_names:
        raw = load_cached_or_remote(
            sheet_id=config.sheet_id,
            sheet_name=sheet,
            cache_dir=config.cache_dir,
            force_refresh=config.force_refresh,
        )
        std = standardize_pede_sheet(raw, sheet_name=sheet)
        frames.append(std)
    df = pd.concat(frames, ignore_index=True, sort=False)

    # Ordenação temporal e deduplicação simples
    if "ra" in df.columns:
        df = df.sort_values(["ra", "ano_referencia"], na_position="last")
        df = df.drop_duplicates(subset=["ra", "ano_referencia"], keep="last")
    else:
        df = df.sort_values(["ano_referencia"], na_position="last")

    # Targets de risco atuais
    if "defasagem" in df.columns:
        df["risco_defasagem_atual"] = (pd.to_numeric(df["defasagem"], errors="coerce") <= -1).astype("Int64")
        df["risco_defasagem_severo"] = (pd.to_numeric(df["defasagem"], errors="coerce") <= -2).astype("Int64")
    elif "ian" in df.columns:
        # fallback heurístico: quartil inferior = risco
        ian_num = pd.to_numeric(df["ian"], errors="coerce")
        q = ian_num.quantile(0.25)
        df["risco_defasagem_atual"] = (ian_num <= q).astype("Int64")
        df["risco_defasagem_severo"] = (ian_num <= ian_num.quantile(0.10)).astype("Int64")

    # Label prospectivo (piora no ano seguinte)
    if {"ra", "ano_referencia"}.issubset(df.columns) and "defasagem" in df.columns:
        df["defasagem_next"] = (
            df.sort_values(["ra", "ano_referencia"])
              .groupby("ra")["defasagem"]
              .shift(-1)
        )
        df["piora_defasagem_prox_ano"] = (
            (pd.to_numeric(df["defasagem_next"], errors="coerce") <
             pd.to_numeric(df["defasagem"], errors="coerce"))
        ).astype("Int64")
    else:
        df["defasagem_next"] = pd.NA
        df["piora_defasagem_prox_ano"] = pd.NA

    return df.reset_index(drop=True)


def summarize_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        rows.append({
            "coluna": c,
            "dtype": str(df[c].dtype),
            "nulos": int(df[c].isna().sum()),
            "pct_nulos": float(df[c].isna().mean()),
            "n_unicos": int(df[c].nunique(dropna=True)),
        })
    return pd.DataFrame(rows).sort_values(["pct_nulos", "n_unicos"], ascending=[False, True])


def get_indicator_columns(df: pd.DataFrame) -> list[str]:
    candidates = ["inde", "ida", "ieg", "iaa", "ips", "ipp", "ipv", "ian"]
    return [c for c in candidates if c in df.columns]


def phase_order_key(series: pd.Series) -> pd.Series:
    # Ordem didática para fases
    mapping = {
        "ALFA": 0, "ALFA (1° E 2° ANO)": 0,
        "FASE 1": 1, "FASE 1 (3° E 4° ANO)": 1,
        "FASE 2": 2, "FASE 2 (5° E 6° ANO)": 2,
        "FASE 3": 3, "FASE 3 (7° E 8° ANO)": 3,
        "FASE 4": 4, "FASE 4 (9° ANO)": 4,
        "FASE 5": 5, "ENSINO MEDIO": 5, "EM": 5
    }
    norm = (
        series.astype(str)
        .str.upper()
        .str.strip()
        .str.replace("Á", "A", regex=False)
        .str.replace("Ç", "C", regex=False)
    )
    return norm.map(mapping).fillna(99)


def add_phase_order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "fase" in df.columns:
        df["fase_ordem"] = phase_order_key(df["fase"])
    else:
        df["fase_ordem"] = np.nan
    return df


def make_long_for_trend(df: pd.DataFrame) -> pd.DataFrame:
    inds = [c for c in ["inde", "ida", "ieg", "iaa", "ips", "ipp", "ipv", "ian"] if c in df.columns]
    id_vars = [c for c in ["ra", "ano_referencia", "fase", "fase_ordem"] if c in df.columns]
    if not inds:
        return pd.DataFrame()
    long_df = df.melt(id_vars=id_vars, value_vars=inds, var_name="indicador", value_name="valor")
    long_df["valor"] = pd.to_numeric(long_df["valor"], errors="coerce")
    return long_df
