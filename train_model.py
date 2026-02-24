from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.passos_data import LoadConfig, add_phase_order, load_all_years
from src.model_passos import TrainConfig, save_model_bundle, train_risk_model


def main():
    parser = argparse.ArgumentParser(description="Treina modelo de risco de defasagem (Passos Mágicos)")
    parser.add_argument("--sheet-id", default="1td91KoeSgXrUrCVOUkLmONG9Go3LVcXpcNEw_XrL2R0")
    parser.add_argument("--sheets", nargs="*", default=["PEDE2022", "PEDE2023", "PEDE2024"])
    parser.add_argument("--cache-dir", default="data_raw")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--target", default="auto", help="auto | risco_defasagem_atual | risco_defasagem_severo | piora_defasagem_prox_ano")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--outdir", default="artifacts")
    args = parser.parse_args()

    cfg = LoadConfig(
        sheet_id=args.sheet_id,
        sheet_names=tuple(args.sheets),
        cache_dir=args.cache_dir,
        force_refresh=args.force_refresh,
    )

    print("[1/4] Carregando e consolidando dados...")
    df = load_all_years(cfg)
    df = add_phase_order(df)
    print(f"Base consolidada: {df.shape[0]} linhas x {df.shape[1]} colunas")

    print("[2/4] Treinando modelo...")
    result = train_risk_model(df, config=TrainConfig(target_col=args.target, threshold=args.threshold))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[3/4] Salvando artefatos...")
    model_path = save_model_bundle(result, outdir / "model_bundle.joblib")

    scored = result["scored_labeled"].copy()
    # tenta anexar identificadores úteis da base original (mesmo índice filtrado pode divergir, então concat por índice)
    id_cols = [c for c in ["ra", "ano_referencia", "fase", "pedra", "genero"] if c in df.columns]
    if id_cols:
        scored = df.loc[scored.index, id_cols].join(scored)

    scored = scored.sort_values("prob_risco", ascending=False)
    scored.to_csv(outdir / "scored_labeled_students.csv", index=False)

    fi = result["feature_importance"].copy()
    fi.to_csv(outdir / "feature_importance.csv", index=False)

    print("[4/4] Resumo")
    print(f"Modelo escolhido: {result['best_model_name']}")
    print(f"Target: {result['target_col']}")
    print(f"Split: {result['split']}")
    print("Métricas (teste):")
    for k, v in result["best_metrics"].items():
        if isinstance(v, (int, float)):
            print(f"  - {k}: {v}")
    print(f"Artefato salvo em: {model_path}")
    print(f"Score CSV salvo em: {outdir / 'scored_labeled_students.csv'}")
    print(f"Importâncias salvas em: {outdir / 'feature_importance.csv'}")


if __name__ == "__main__":
    main()
