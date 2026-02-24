# Roteiro sugerido de apresentação (PPT/PDF)

## Slide 1 — Capa
- Datathon Passos Mágicos
- Equipe / integrantes
- Objetivo: identificar padrões e antecipar risco de defasagem

## Slide 2 — Contexto e problema
- Missão da ONG
- Desafio analítico: transformar indicadores em decisão preventiva
- Período analisado: 2022–2024

## Slide 3 — Base de dados e metodologia
- Abas PEDE2022/2023/2024
- Etapas: ingestão, limpeza, padronização, análise, ML, app
- Observações de qualidade dos dados (faltantes, colunas heterogêneas)

## Slide 4 — Q1 (IAN / defasagem)
- Perfil de defasagem por ano (moderada/severa)
- Evolução temporal
- Insight acionável (priorização de acompanhamento)

## Slide 5 — Q2 + Q10 (IDA e efetividade por fase)
- Tendência do IDA ao longo dos anos
- Comparação por fase (Quartzo/Ágata/Ametista/Topázio, se disponível)
- Sinais de melhora/estagnação

## Slide 6 — Q3 + Q4 (IEG, IPV, IAA)
- Relações IEG x IDA / IEG x IPV
- Coerência IAA vs desempenho real
- Segmentos com desalinhamento percepção x realidade

## Slide 7 — Q5 + Q6 (IPS e IPP)
- Padrões psicossociais e psicopedagógicos
- Concordância/contradição com defasagem (IAN)
- Oportunidades de intervenção multidisciplinar

## Slide 8 — Q7 + Q8 (drivers do IPV e INDE)
- Matriz de correlação
- Combinações que mais elevam INDE
- Fatores-chave priorizados

## Slide 9 — Q9 (Modelo preditivo de risco)
- Target utilizado (auto -> preferencialmente piora futura)
- Feature engineering
- Split treino/teste (temporal quando possível)
- Métricas (ROC-AUC / AP)
- Interpretação dos principais drivers

## Slide 10 — Solução aplicada (Streamlit)
- Screenshot da aplicação
- Ranking de risco + predição manual + upload CSV
- Como a equipe pedagógica pode usar no dia a dia

## Slide 11 — Recomendações práticas
- Threshold por capacidade de atendimento
- Rotina de reavaliação mensal/bimestral
- Integração com acompanhamento psicopedagógico
- Monitoramento de qualidade de dados

## Slide 12 — Próximos passos
- Re-treino com novas safras
- Calibração e fairness por subgrupos
- Acompanhamento de impacto das intervenções
