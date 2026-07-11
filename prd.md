# SemanticOps — Product Requirements Document

**Author:** Portfolio Project  
**Inspired by:** LLM-Guided Task-Semantic Field Factorization for Industrial Process Forecasting

---

## 1. Problem Statement

Industrial forecasting datasets from manufacturing, energy, and chemical process industries share a common problem: column names are meaningless codes (`TAG_001`, `column_0`), documentation is sparse, and the physical relationships between variables are implicit domain knowledge that sits in engineers' heads — not in the data.

This means standard AutoML / feature engineering tools perform poorly because they cannot leverage domain semantics.

## 2. Vision

SemanticOps is the **industrial forecasting agent** that reads messy data the way a domain expert would: understanding what each sensor measures, its units, its causal role, and how it relates to the target. This semantic understanding is then encoded into feature engineering that outperforms blind tabular AutoML.

## 3. Core Features

| Feature | Description |
|---|---|
| Schema agent | LLM interprets column names → semantic name, units, role (target/feature/timestamp) |
| Feature extractor | Lag features, rolling stats, cyclic encoding — all semantically informed |
| Forecaster | Gradient boosting with TimeSeriesSplit CV + feature importance |
| Explanation | Plain-language explanation of which variables drive predictions |
| FastAPI | `/upload`, `/schema`, `/explain`, `/predict` endpoints |
| Heuristic fallback | Full offline operation without API key |

## 4. Implementation Plan

### Phase 1 — Schema Agent (complete)
- `SchemaAgent`: Claude-powered schema interpreter with heuristic fallback
- Detects target, timestamp, feature, and identifier columns
- Returns units, semantic names, cyclic flags

### Phase 2 — Feature Engine (complete)
- `FeatureExtractor`: lag features, rolling stats, cyclic encoding, datetime features
- Guided by schema metadata for informed feature selection

### Phase 3 — Forecaster (complete)
- `Forecaster`: GBM with TimeSeriesSplit CV
- Feature importance ranking
- Plain-language prediction explanation

### Phase 4 — Integration (complete)
- `SemanticOpsPipeline`: full orchestration
- FastAPI server with file upload, schema, explain, predict endpoints

### Phase 5 — Advanced
- LangGraph agent loop: schema → query clarification → feature iteration
- TimesFM / Chronos neural forecaster backend
- SHAP-based deep explanations
- MLflow experiment tracking
- React dashboard: schema viewer, prediction chart, feature importance heatmap
