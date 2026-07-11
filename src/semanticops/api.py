"""
FastAPI server for SemanticOps.

Exposes:
  POST /upload      — upload CSV + documentation, run full pipeline
  POST /predict     — run inference on new data rows
  GET  /schema      — retrieve last interpreted schema
  GET  /model       — training report for current model
  GET  /explain     — feature importance explanation
  GET  /health      — liveness probe

Run with:
    uvicorn semanticops.api:app --host 0.0.0.0 --port 8002 --reload
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from semanticops.pipeline import SemanticOpsPipeline, PipelineResult

app = FastAPI(
    title="SemanticOps",
    description="Industrial AI Agent — LLM-guided forecasting for messy process data",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = SemanticOpsPipeline()
_last_result: PipelineResult | None = None


# --- Request / Response models ---

class PredictRequest(BaseModel):
    rows: list[dict[str, Any]]   # new data rows as list of dicts


# --- Endpoints ---

@app.get("/health")
def health() -> dict[str, str]:
    trained = _last_result is not None and _last_result.success
    return {"status": "ok", "model_trained": str(trained)}


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    documentation: str = Form(default=""),
    target_column: str = Form(default=""),
) -> dict[str, Any]:
    """Upload a CSV file and run the full SemanticOps pipeline."""
    global _last_result

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    target = target_column.strip() or None
    result = _pipeline.run_from_dataframe(df, documentation=documentation, target_column=target)
    _last_result = result

    if not result.success:
        raise HTTPException(status_code=422, detail=result.error)

    return _pipeline.to_api_response(result)


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    """Run inference on new data rows."""
    if _last_result is None or not _last_result.success:
        raise HTTPException(status_code=400, detail="No trained model. POST to /upload first.")
    try:
        df = pd.DataFrame(req.rows)
        predictions = _pipeline.predict_new(df)
        return {
            "predictions": predictions.tolist(),
            "target": _last_result.schema.target_column,
            "n_rows": len(predictions),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/schema")
def get_schema() -> dict[str, Any]:
    if _last_result is None:
        raise HTTPException(status_code=400, detail="No data uploaded yet.")
    schema = _last_result.schema
    return {
        "domain": schema.domain,
        "summary": schema.summary,
        "target_column": schema.target_column,
        "feature_columns": schema.feature_columns,
        "columns": [
            {
                "name": c.name,
                "semantic_name": c.semantic_name,
                "role": c.role.value,
                "units": c.units,
                "description": c.description,
            }
            for c in schema.columns
        ],
    }


@app.get("/model")
def get_model() -> dict[str, Any]:
    if _last_result is None or not _last_result.success:
        raise HTTPException(status_code=400, detail="No trained model yet.")
    tr = _last_result.training_report
    return {
        "model_type": tr.model_type,
        "n_train": tr.n_train,
        "n_val": tr.n_val,
        "val_mae": tr.val_mae,
        "val_rmse": tr.val_rmse,
        "val_r2": tr.val_r2,
        "cv_scores": tr.cv_scores,
        "summary": tr.summary,
    }


@app.get("/explain")
def explain() -> dict[str, Any]:
    if _last_result is None or not _last_result.success:
        raise HTTPException(status_code=400, detail="No trained model yet.")
    sf = _last_result.sample_forecast
    return {
        "explanation": sf.explanation,
        "top_features": sf.top_features,
        "schema_context": {
            col.name: {
                "semantic_name": col.semantic_name,
                "units": col.units,
                "description": col.description,
            }
            for col in _last_result.schema.columns
        },
    }
