from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.passos_data import LoadConfig, add_phase_order, get_indicator_columns, load_all_years
from src.model_passos import (
    TrainConfig,
    load_model_bundle,
    save_model_bundle,
    score_dataframe,
    train_risk_model,
)

st.set_page_config(page_title="Passos Mágicos — Risco de Defasagem", page_icon="📚", layout="wide")

DEFAULT_SHEET_ID = "1td91KoeSgXrUrCVOUkLmONG9Go3LVcXpcNEw_XrL2R0"
MODEL_PATH = Path("artifacts/model_bundle.joblib")


@st.cache_data(show_spinner=False)
def load_data(sheet_id: str, force_refresh: bool = False) -> pd.DataFrame:
    cfg = LoadConfig(
        sheet_id=sheet_id,
        sheet_names=("PEDE2022", "PEDE2023", "PEDE2024"),
        cache_dir="data_raw",
        force_refresh=force_refresh,
    )
    df = load_all_years(cfg)
    df = add_phase_order(df)
    return df


@st.cache_resource(show_spinner=False)
def train_or_load_bundle(df: pd.DataFrame, target: str = "auto", threshold: float = 0.50, retrain: bool = False):
    if MODEL_PATH.exists() and not retrain:
        return load_model_bundle(MODEL_PATH), None

    result = train_risk_model(df, config=TrainConfig(target_col=target, threshold=threshold))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_model_bundle(result, MODEL_PATH)
    return load_model_bundle(MODEL_PATH), result


def main():
    st.title("📚 Passos Mágicos — Predição de Risco de Defasagem")
    st.caption("Aplicação Streamlit para apoiar identificação antecipada de risco com base nos indicadores educacionais.")

    with st.sidebar:
        st.header("Configurações")
        sheet_id = st.text_input("Google Sheet ID", value=DEFAULT_SHEET_ID)
        threshold = st.slider("Threshold de classificação", 0.10, 0.90, 0.50, 0.01)
        target_mode = st.selectbox(
            "Target do modelo",
            options=["auto", "piora_defasagem_prox_ano", "risco_defasagem_atual", "risco_defasagem_severo"],
            index=0,
            help="'auto' escolhe o melhor target disponível no dataset (prioriza piora no próximo ano se houver rótulos suficientes).",
        )
        col1, col2 = st.columns(2)
        with col1:
            force_refresh = st.checkbox("Recarregar planilha", value=False)
        with col2:
            retrain = st.checkbox("Retreinar modelo", value=False)

    try:
        with st.spinner("Carregando dados..."):
            df = load_data(sheet_id=sheet_id, force_refresh=force_refresh)
    except Exception as e:
        st.error(f"Erro ao carregar dados da planilha pública: {e}")
        st.stop()

    st.success(f"Base carregada: {df.shape[0]} linhas × {df.shape[1]} colunas")

    try:
        with st.spinner("Carregando/treinando modelo..."):
            bundle, train_result = train_or_load_bundle(df, target=target_mode, threshold=threshold, retrain=retrain)
    except Exception as e:
        st.error(f"Erro no treino/carregamento do modelo: {e}")
        st.stop()

    st.subheader("Resumo do modelo")
    m1, m2, m3, m4 = st.columns(4)
    best_metrics = bundle.get("best_metrics", {})
    with m1:
        st.metric("Target", bundle.get("target_col", "-"))
    with m2:
        roc = best_metrics.get("roc_auc")
        st.metric("ROC-AUC (teste)", "-" if roc is None or pd.isna(roc) else f"{roc:.3f}")
    with m3:
        ap = best_metrics.get("average_precision")
        st.metric("AP (teste)", "-" if ap is None or pd.isna(ap) else f"{ap:.3f}")
    with m4:
        st.metric("Threshold", f"{bundle.get('threshold', threshold):.2f}")

    split = bundle.get("split", {})
    with st.expander("Detalhes do treino e split"):
        st.json({
            "best_model_name": bundle.get("best_model_name"),
            "split": split,
            "schema": bundle.get("schema", {}),
            "best_metrics": best_metrics,
        })

    # Score da base atual
    st.subheader("Ranking de risco na base")
    try:
        scored = score_dataframe(df, bundle)
    except Exception as e:
        st.error(f"Falha ao pontuar a base: {e}")
        st.stop()

    # Colunas úteis para exibição
    show_cols = [c for c in ["ra", "ano_referencia", "fase", "pedra", "genero", "inde", "ida", "ieg", "iaa", "ips", "ipp", "ipv"] if c in scored.columns]
    scored = scored.copy()
    scored["prob_risco"] = pd.to_numeric(scored["prob_risco"], errors="coerce")
    scored["pred_risco"] = (scored["prob_risco"] >= float(bundle.get("threshold", threshold))).astype(int)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Alunos pontuados", int(scored["prob_risco"].notna().sum()))
    with c2:
        st.metric("Risco alto (pred=1)", int(scored["pred_risco"].sum()))
    with c3:
        st.metric("Prob. média de risco", f"{scored['prob_risco'].mean():.2%}")

    tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "🧪 Predição manual", "📤 Upload CSV"])

    with tab1:
        f1, f2, f3 = st.columns(3)
        with f1:
            anos = sorted([a for a in pd.to_numeric(scored.get("ano_referencia"), errors="coerce").dropna().unique()]) if "ano_referencia" in scored.columns else []
            ano_sel = st.selectbox("Filtrar ano", options=["Todos"] + [int(a) for a in anos])
        with f2:
            fase_opts = ["Todas"]
            if "fase" in scored.columns:
                fase_opts += sorted([str(x) for x in scored["fase"].dropna().astype(str).unique()])
            fase_sel = st.selectbox("Filtrar fase", options=fase_opts)
        with f3:
            top_n = st.slider("Top N alunos", 10, 200, 30, 10)

        view = scored.copy()
        if ano_sel != "Todos" and "ano_referencia" in view.columns:
            view = view[pd.to_numeric(view["ano_referencia"], errors="coerce") == int(ano_sel)]
        if fase_sel != "Todas" and "fase" in view.columns:
            view = view[view["fase"].astype(str) == fase_sel]

        # Gráfico de distribuição simples
        st.markdown("**Distribuição da probabilidade de risco**")
        hist_vals = view["prob_risco"].dropna().values
        if len(hist_vals) > 0:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(hist_vals, bins=20)
            ax.set_xlabel("Probabilidade de risco")
            ax.set_ylabel("Quantidade de alunos")
            ax.axvline(float(bundle.get("threshold", threshold)), linestyle="--")
            ax.grid(alpha=0.2, axis="y")
            st.pyplot(fig)
        else:
            st.info("Sem dados para o filtro selecionado.")

        st.markdown("**Top alunos por probabilidade de risco**")
        cols_table = show_cols + ["prob_risco", "pred_risco"]
        top_df = view.sort_values("prob_risco", ascending=False)[cols_table].head(top_n)
        st.dataframe(top_df, use_container_width=True, hide_index=True)
        st.download_button(
            label="Baixar ranking filtrado (CSV)",
            data=top_df.to_csv(index=False).encode("utf-8"),
            file_name="ranking_risco_filtrado.csv",
            mime="text/csv",
        )

        if "fase" in view.columns:
            st.markdown("**Probabilidade média por fase**")
            grp = (view.groupby("fase", dropna=False)["prob_risco"].mean().sort_values(ascending=False))
            st.bar_chart(grp)

        if "ano_referencia" in view.columns:
            st.markdown("**Probabilidade média por ano**")
            grp_year = (view.groupby("ano_referencia", dropna=False)["prob_risco"].mean().sort_index())
            st.line_chart(grp_year)

    with tab2:
        st.markdown("Informe indicadores de um aluno para estimar a probabilidade de risco.")
        schema = bundle.get("schema", {})
        numeric_cols = schema.get("numeric", [])
        categorical_cols = schema.get("categorical", [])

        defaults = {}
        for c in numeric_cols:
            defaults[c] = float(pd.to_numeric(df[c], errors="coerce").median()) if c in df.columns else 0.0
        for c in categorical_cols:
            vals = []
            if c in df.columns:
                vals = [v for v in df[c].dropna().astype(str).unique().tolist() if str(v).strip()]
            defaults[c] = vals[0] if vals else ""

        with st.form("manual_pred_form"):
            inputs = {}
            cols_layout = st.columns(2)
            i = 0
            for c in numeric_cols:
                with cols_layout[i % 2]:
                    inputs[c] = st.number_input(c.upper(), value=float(defaults.get(c, 0.0)))
                i += 1
            for c in categorical_cols:
                with cols_layout[i % 2]:
                    options = [str(v) for v in df[c].dropna().astype(str).unique().tolist()] if c in df.columns else []
                    options = options or [""]
                    default_val = str(defaults.get(c, options[0]))
                    if default_val not in options:
                        options = [default_val] + options
                    inputs[c] = st.selectbox(c, options=options, index=options.index(default_val))
                i += 1
            submitted = st.form_submit_button("Calcular risco")

        if submitted:
            row = pd.DataFrame([inputs])
            scored_manual = score_dataframe(row, bundle)
            p = float(scored_manual.loc[0, "prob_risco"])
            pred = int(scored_manual.loc[0, "pred_risco"])
            st.metric("Probabilidade de risco", f"{p:.2%}")
            st.metric("Classificação", "Risco alto" if pred == 1 else "Risco baixo")
            st.progress(min(max(p, 0.0), 1.0))

    with tab3:
        st.markdown("Faça upload de um CSV com colunas compatíveis (ex.: IDA, IEG, IAA, IPS, IPP, IPV, fase, gênero...).")
        up = st.file_uploader("CSV para pontuar", type=["csv"])
        if up is not None:
            try:
                up_df = pd.read_csv(up)
                up_scored = score_dataframe(up_df, bundle)
                st.dataframe(up_scored.head(50), use_container_width=True)
                st.download_button(
                    "Baixar CSV pontuado",
                    data=up_scored.to_csv(index=False).encode("utf-8"),
                    file_name="csv_pontuado_risco.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Erro ao pontuar arquivo enviado: {e}")

    # Insights operacionais rápidos
    st.subheader("Sugestões de uso operacional")
    st.markdown(
        "- Use o ranking como **triagem inicial**, não como decisão final.\n"
        "- Combine a probabilidade com contexto qualitativo (psicopedagógico/psicossocial).\n"
        "- Recalibre o threshold conforme capacidade de atendimento (ex.: top 10%, top 20%).\n"
        "- Re-treine periodicamente ao entrar nova safra/ano."
    )


if __name__ == "__main__":
    main()
