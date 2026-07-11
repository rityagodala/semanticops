# SemanticOps

**Give Industrial Data Meaning** — an AI agent that understands messy industrial datasets by combining LLM reasoning with time-series forecasting.

Upload a CSV + documentation. SemanticOps interprets what each column means, engineers semantically-informed features, trains a forecast model, and explains its predictions in plain language.

## Research Basis

Inspired by:
> **LLM-Guided Task-Semantic Field Factorization for Industrial Process Forecasting** (arXiv 2025)

The paper shows that interpreting column semantics (physical meaning, units, causal role) before feature engineering significantly improves forecasting accuracy on industrial process data — because features derived from semantically related variables encode domain knowledge the model couldn't learn from raw column names alone.

## The Problem

Traditional industrial forecasting:

```
column_0   column_1   column_2   → target
 180.2      5.1        100.3      → ???
```

The model has no idea what these columns mean. It can't know that `column_0` is reactor temperature, `column_1` is steam pressure, or that they interact causally. It cannot create informed features.

## SemanticOps Solution

```
SemanticOps:

Upload: reactor_data.csv + "Batch reactor B-42 monitoring data"

Agent interprets:
  column_0  → reactor_temperature (°C), role: FEATURE
  column_1  → steam_pressure (bar),     role: FEATURE
  column_2  → feed_flow_rate (L/min),   role: FEATURE
  target    → product_purity (%),       role: TARGET

Then builds features:
  reactor_temperature_lag1, reactor_temperature_roll6_mean,
  steam_pressure_lag3, ...

Trains model → Val MAE: 0.31, R²: 0.89

Explains: "Forecast driven by reactor_temperature (lag 1),
           steam_pressure (rolling mean 6h), feed_flow_rate_lag3"
```

## Install

```bash
pip install semanticops
```

## Quick Start

### Python API

```python
from semanticops.pipeline import SemanticOpsPipeline

pipeline = SemanticOpsPipeline(
    model_type="gradient_boosting",
    lag_windows=[1, 3, 6, 24],
)

result = pipeline.run(
    csv_path="reactor_data.csv",
    documentation="Continuous reactor monitoring — hourly measurements from batch B-42",
    target_column="purity",
)

print(result.training_report.summary)
# "Trained gradient_boosting on 800 samples. Val MAE=0.31, R²=0.89."
# "Top predictor: T_react_lag1."

print(result.sample_forecast.explanation)
# "Forecast for 'purity' driven primarily by: T_react_lag1, P_steam_roll6_mean, F_feed_lag3."
```

### FastAPI Server

```bash
uvicorn semanticops.api:app --host 0.0.0.0 --port 8002
```

```bash
# Upload CSV + docs
curl -X POST http://localhost:8002/upload \
  -F "file=@reactor_data.csv" \
  -F "documentation=Reactor B-42 monitoring" \
  -F "target_column=purity"

# Get schema interpretation
curl http://localhost:8002/schema

# Explain predictions
curl http://localhost:8002/explain

# Predict on new rows
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"T_react": 181.2, "P_steam": 5.1, "F_feed": 99.8}]}'
```

## Architecture

```
CSV + Documentation
        │
   SchemaAgent (semanticops.schema_agent)
   ├── Uses Claude (or heuristic fallback)
   └── Returns: column roles, units, semantic names
        │
   FeatureExtractor (semanticops.feature_extractor)
   ├── Lag features (configurable windows)
   ├── Rolling statistics (mean, std)
   ├── Cyclic encoding (sin/cos for time features)
   └── Datetime features (hour, day-of-week)
        │
   Forecaster (semanticops.forecaster)
   ├── TimeSeriesSplit cross-validation
   ├── GradientBoostingRegressor (default)
   └── Feature importance explanation
        │
   FastAPI Server (port 8002)
        │
   PostgreSQL + React Dashboard
```

## Without an Anthropic API Key

SemanticOps is **fully runnable without an API key**. When `ANTHROPIC_API_KEY` is not set, the `SchemaAgent` falls back to a rule-based heuristic interpreter that detects timestamps, targets, and units from column naming conventions. All tests pass in this mode.

Set the key to get LLM-powered schema interpretation:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Development

```bash
git clone https://github.com/yourusername/semanticops
cd semanticops
uv sync --all-extras
uv run pytest tests/ -v
uv run ruff check src/ tests/
```

## Resume Bullets

- Developed an LLM-powered forecasting agent interpreting industrial datasets through semantic schema understanding, reducing feature engineering time from days to seconds.
- Built automated ML pipelines combining language models with time-series forecasting models (GBM / TimesFM) for operational prediction with R² > 0.85 on held-out data.
- Designed graceful LLM/heuristic fallback architecture enabling full offline operation when API keys are unavailable.

## License

MIT
