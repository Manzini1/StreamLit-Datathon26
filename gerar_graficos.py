from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from passos_data import LoadConfig, load_all_years, add_phase_order, make_long_for_trend
from plots_passos import (
    plot_barras_defasagem_por_ano,
    plot_serie_indicadores,
    plot_matriz_correlacao,
    plot_scatter_relacao,
    plot_box_indicador_por_fase,
)

def main():
    out_figs = PROJECT_ROOT / "outputs" / "figs"
    out_figs.mkdir(parents=True, exist_ok=True)

    csv_path = PROJECT_ROOT / "data_processed" / "pede_consolidado.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        cfg = LoadConfig(
            sheet_id="1td91KoeSgXrUrCVOUkLmONG9Go3LVcXpcNEw_XrL2R0",
            sheet_names=("PEDE2022", "PEDE2023", "PEDE2024"),
            cache_dir=PROJECT_ROOT / "data_raw",
            force_refresh=False,
        )
        df = load_all_years(cfg)
        df.to_csv(csv_path, index=False)

    df = add_phase_order(df)
    for c in ["inde","ida","ieg","iaa","ips","ipp","ipv","ian","defasagem"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Gráficos macro
    if "defasagem" in df.columns:
        plot_barras_defasagem_por_ano(df, outpath=out_figs / "defasagem_por_ano.png", show=False)

    long_df = make_long_for_trend(df)
    if not long_df.empty:
        plot_serie_indicadores(long_df, indicadores=["inde","ida","ieg","iaa","ips","ipp","ipv","ian"], outpath=out_figs / "tendencia_indicadores.png", show=False)

    corr_cols = [c for c in ["inde","ida","ieg","iaa","ips","ipp","ipv","ian","defasagem"] if c in df.columns]
    if len(corr_cols) >= 2:
        plot_matriz_correlacao(df, corr_cols, outpath=out_figs / "matriz_correlacao.png", show=False)

    # Relações chave
    for x, y in [("ieg","ida"), ("ieg","ipv"), ("iaa","ida"), ("ipp","ian"), ("ips","ida")]:
        if x in df.columns and y in df.columns:
            plot_scatter_relacao(df, x, y, hue="ano_referencia" if "ano_referencia" in df.columns else None,
                                 outpath=out_figs / f"scatter_{x}_vs_{y}.png", show=False)

    # Boxplots por fase
    if "fase" in df.columns:
        for indicador in ["inde","ida","ieg","ipv","ian"]:
            if indicador in df.columns:
                try:
                    plot_box_indicador_por_fase(df, indicador, outpath=out_figs / f"box_{indicador}_por_fase.png", show=False)
                except Exception:
                    pass

    print(f"Gráficos gerados em: {out_figs}")

if __name__ == "__main__":
    main()
