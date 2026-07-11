# SemanticOps Architecture

## Pipeline Flow

```
CSV Upload + Documentation
          
    ┌─────▼───────────────────────────────┐
    │           SchemaAgent               │
    │                                     │
    │  Column names + docs + sample vals  │
    │              ↓                      │
    │  Claude (or heuristic fallback)     │
    │              ↓                      │
    │  SchemaInterpretation               │
    │  • semantic_name per column         │
    │  • role: target/feature/timestamp   │
    │  • units, cyclic flag               │
    └─────┬───────────────────────────────┘
          │
    ┌─────▼───────────────────────────────┐
    │         FeatureExtractor            │
    │                                     │
    │  • Lag features (configurable)      │
    │  • Rolling mean / std               │
    │  • Sin/cos for cyclic columns       │
    │  • Datetime features                │
    └─────┬───────────────────────────────┘
          │
    ┌─────▼───────────────────────────────┐
    │            Forecaster               │
    │                                     │
    │  • TimeSeriesSplit CV               │
    │  • GradientBoostingRegressor        │
    │  • Feature importance ranking       │
    │  • Plain-language explanation       │
    └─────┬───────────────────────────────┘
          │
    FastAPI (port 8002)
    /upload  /schema  /explain  /predict

```

## Schema Agent: LLM vs Heuristic

When ANTHROPIC_API_KEY is set:
  → Claude interprets column meanings from names + docs + sample values
  → Returns structured JSON with semantic metadata

When no API key:
  → Rule-based heuristic detects common patterns:
    "timestamp", "date" → TIMESTAMP
    "purity", "yield", "quality", "target" → TARGET
    "id", "batch", "lot" → IDENTIFIER
    everything else → FEATURE
    temperature keywords → units: °C
    pressure keywords → units: bar
