# Kit Python — Datathon Passos Mágicos

Este kit foi montado para você rodar **localmente** e gerar:
- base consolidada (PEDE2022 + PEDE2023 + PEDE2024)
- análises das perguntas do datathon
- modelo preditivo de risco de defasagem
- gráficos em PNG para usar na apresentação

## Estrutura
- `00_ingestao_qualidade.ipynb`
- `01_analise_negocio_perguntas.ipynb`
- `02_modelo_risco_ml.ipynb`
- `gerar_graficos.py`
- `src/passos_data.py`
- `src/plots_passos.py`
- `requirements.txt`

## Como rodar (Windows / VS Code)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

### Ordem recomendada
1. Execute `00_ingestao_qualidade.ipynb`
2. Execute `01_analise_negocio_perguntas.ipynb`
3. Execute `02_modelo_risco_ml.ipynb`

### Gerar gráficos via script (sem notebook)
```bash
python gerar_graficos.py
```

## Observações importantes
- O código tenta ler as abas públicas via Google Sheets (`gviz/tq?...out:csv`).
- Se alguma aba mudar de nome, ajuste em `LoadConfig(... sheet_names=...)`.
- Se a planilha mudar estrutura de colunas, a padronização já é tolerante, mas pode precisar de pequenos ajustes de aliases em `src/passos_data.py`.

## Entregáveis para a apresentação gerencial
- Gráficos de tendência (IDA/INDE/IEG/IPV/IAN)
- Perfil de defasagem por ano/fase
- Correlações e relações entre indicadores
- Tabela de alunos com probabilidade de risco
- Top fatores explicativos do modelo

## Novo: Aplicação Streamlit (predição de risco)
Arquivos adicionados:
- `app.py` (interface Streamlit)
- `train_model.py` (treino e geração de artefatos)
- `src/model_passos.py` (pipeline de ML reutilizável)

### Treinar e salvar o modelo localmente
```bash
python train_model.py --target auto --threshold 0.50
```
Isso gera:
- `artifacts/model_bundle.joblib`
- `artifacts/scored_labeled_students.csv`
- `artifacts/feature_importance.csv`

### Rodar a aplicação Streamlit localmente
```bash
streamlit run app.py
```

## Deploy no Streamlit Community Cloud
1. Suba este projeto no GitHub
2. No Streamlit Community Cloud, clique em **New app**
3. Selecione o repositório e o arquivo `app.py`
4. Confirme o deploy

> Dica: se quiser evitar treino no primeiro acesso, rode `train_model.py` localmente e suba a pasta `artifacts/` com `model_bundle.joblib` para o repositório.
