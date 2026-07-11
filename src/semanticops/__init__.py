"""
SemanticOps: Industrial AI Agent.

Combines LLM semantic understanding of messy industrial datasets with
time-series forecasting. Upload a CSV + documentation; the agent
interprets the schema, engineers features, trains a forecast model,
and explains its predictions.
"""

from semanticops.schema_agent import SchemaAgent
from semanticops.feature_extractor import FeatureExtractor
from semanticops.forecaster import Forecaster
from semanticops.pipeline import SemanticOpsPipeline

__version__ = "0.1.0"
__all__ = ["SchemaAgent", "FeatureExtractor", "Forecaster", "SemanticOpsPipeline"]
