from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_outpath(outpath: str | Path | None):
    if outpath is None:
        return None
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    return outpath


def _finalize(outpath=None, show=True):
    if outpath is not None:
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_distribuicao_defasagem(df: pd.DataFrame, outpath=None, show=True):
    if "defasagem" not in df.columns:
        raise ValueError("Coluna 'defasagem' não encontrada.")
    s = pd.to_numeric(df["defasagem"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(np.floor(s.min()) - 0.5, np.ceil(s.max()) + 1.5, 1)
    ax.hist(s, bins=bins)
    ax.set_title("Distribuição de Defasagem")
    ax.set_xlabel("Defasagem")
    ax.set_ylabel("Quantidade de alunos")
    ax.grid(alpha=0.2, axis="y")
    _finalize(_prepare_outpath(outpath), show)


def plot_barras_defasagem_por_ano(df: pd.DataFrame, outpath=None, show=True):
    if not {"ano_referencia", "defasagem"}.issubset(df.columns):
        raise ValueError("Colunas necessárias não encontradas.")
    tmp = df.copy()
    tmp["defasagem"] = pd.to_numeric(tmp["defasagem"], errors="coerce")
    tmp["categoria_defasagem"] = np.select(
        [tmp["defasagem"] <= -2, tmp["defasagem"] == -1, tmp["defasagem"] >= 0],
        ["Severa (<= -2)", "Moderada (-1)", "Adequado/Acima (>= 0)"],
        default="Sem dado"
    )
    pt = (tmp.groupby(["ano_referencia", "categoria_defasagem"])
            .size()
            .unstack(fill_value=0)
            .sort_index())
    pt = pt[[c for c in ["Severa (<= -2)", "Moderada (-1)", "Adequado/Acima (>= 0)", "Sem dado"] if c in pt.columns]]
    ax = pt.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Perfil de defasagem por ano")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Quantidade")
    ax.legend(title="Categoria", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.2, axis="y")
    _finalize(_prepare_outpath(outpath), show)


def plot_serie_indicadores(long_df: pd.DataFrame, indicadores=None, outpath=None, show=True):
    if long_df.empty:
        raise ValueError("DataFrame long está vazio.")
    plot_df = long_df.copy()
    if indicadores is not None:
        plot_df = plot_df[plot_df["indicador"].isin(indicadores)]
    agg = (plot_df.groupby(["ano_referencia", "indicador"], as_index=False)["valor"]
                 .mean()
                 .dropna())
    fig, ax = plt.subplots(figsize=(10, 5))
    for ind, g in agg.groupby("indicador"):
        ax.plot(g["ano_referencia"], g["valor"], marker="o", label=ind.upper())
    ax.set_title("Evolução média dos indicadores por ano")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Média")
    ax.grid(alpha=0.25)
    ax.legend(ncol=4, bbox_to_anchor=(0.5, -0.15), loc="upper center")
    _finalize(_prepare_outpath(outpath), show)


def plot_scatter_relacao(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None, outpath=None, show=True):
    plot_df = df[[c for c in [x, y, hue] if c is not None and c in df.columns]].copy()
    plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y])
    fig, ax = plt.subplots(figsize=(7, 5))
    if hue and hue in plot_df.columns:
        # Mapeia categorias manualmente sem dependência de seaborn
        for cat, g in plot_df.groupby(hue):
            ax.scatter(g[x], g[y], s=20, alpha=0.7, label=str(cat))
        ax.legend(title=hue, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        ax.scatter(plot_df[x], plot_df[y], s=20, alpha=0.7)
    ax.set_title(f"Relação {x.upper()} vs {y.upper()}")
    ax.set_xlabel(x.upper())
    ax.set_ylabel(y.upper())
    ax.grid(alpha=0.2)
    _finalize(_prepare_outpath(outpath), show)


def plot_matriz_correlacao(df: pd.DataFrame, cols: Iterable[str], outpath=None, show=True):
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        raise ValueError("São necessárias ao menos 2 colunas para correlação.")
    tmp = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = tmp.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels([c.upper() for c in cols], rotation=45, ha="right")
    ax.set_yticklabels([c.upper() for c in cols])
    ax.set_title("Matriz de correlação")
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _finalize(_prepare_outpath(outpath), show)


def plot_box_indicador_por_fase(df: pd.DataFrame, indicador: str, outpath=None, show=True):
    if indicador not in df.columns or "fase" not in df.columns:
        raise ValueError("Colunas necessárias não encontradas.")
    tmp = df[["fase", indicador]].copy()
    tmp[indicador] = pd.to_numeric(tmp[indicador], errors="coerce")
    tmp = tmp.dropna()
    fases = list(tmp["fase"].dropna().astype(str).unique())
    data = [tmp.loc[tmp["fase"] == f, indicador].dropna().values for f in fases]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.boxplot(data, labels=fases, showfliers=False)
    ax.set_title(f"{indicador.upper()} por fase")
    ax.set_xlabel("Fase")
    ax.set_ylabel(indicador.upper())
    plt.xticks(rotation=30, ha="right")
    ax.grid(alpha=0.2, axis="y")
    _finalize(_prepare_outpath(outpath), show)


def plot_feature_importance(importances: pd.Series, outpath=None, show=True, top_n: int = 15):
    s = importances.sort_values(ascending=False).head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(s.index.astype(str), s.values)
    ax.set_title("Importância das variáveis (top)")
    ax.set_xlabel("Importância")
    ax.set_ylabel("Feature")
    ax.grid(alpha=0.2, axis="x")
    _finalize(_prepare_outpath(outpath), show)


def plot_calibration_curve(y_true, y_prob, n_bins=10, outpath=None, show=True):
    y_true = pd.Series(y_true).astype(float)
    y_prob = pd.Series(y_prob).astype(float)
    tmp = pd.DataFrame({"y": y_true, "p": y_prob}).dropna()
    tmp["bin"] = pd.qcut(tmp["p"], q=min(n_bins, tmp["p"].nunique()), duplicates="drop")
    cal = tmp.groupby("bin", observed=True).agg(prob_media=("p", "mean"), taxa_real=("y", "mean"))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(cal["prob_media"], cal["taxa_real"], marker="o")
    ax.set_title("Curva de calibração")
    ax.set_xlabel("Probabilidade prevista média")
    ax.set_ylabel("Taxa real")
    ax.grid(alpha=0.2)
    _finalize(_prepare_outpath(outpath), show)
