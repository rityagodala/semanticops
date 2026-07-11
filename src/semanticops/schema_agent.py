"""
LLM-powered schema interpreter.

The breakthrough insight from the paper:

  Traditional:                        SemanticOps:
  column_0, column_1, column_2   →   reactor_temperature (°C),
  (meaningless to forecaster)         steam_pressure (bar),
                                       product_purity_pct (quality indicator)

The agent reads raw column names + optional documentation and returns
structured metadata about each column: physical meaning, units, role
(target / feature / identifier / timestamp), and quality notes.

Inspired by:
  "LLM-Guided Task-Semantic Field Factorization for
   Industrial Process Forecasting" (arXiv 2025)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    from anthropic import Anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


class ColumnRole(Enum):
    TARGET = "target"           # what we want to forecast
    FEATURE = "feature"         # input to the forecast model
    TIMESTAMP = "timestamp"     # time index
    IDENTIFIER = "identifier"   # ID column (drop before training)
    UNKNOWN = "unknown"


@dataclass
class ColumnMetadata:
    name: str
    semantic_name: str          # human-readable interpretation
    role: ColumnRole
    units: str | None
    description: str
    is_cyclic: bool = False     # e.g. hour-of-day, day-of-week
    quality_notes: str = ""


@dataclass
class SchemaInterpretation:
    """Full schema interpretation for one dataset."""

    columns: list[ColumnMetadata]
    domain: str                 # inferred domain (e.g. "chemical process")
    target_column: str | None
    timestamp_column: str | None
    suggested_features: list[str]
    summary: str
    raw_llm_response: str = ""

    @property
    def feature_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.role == ColumnRole.FEATURE]

    @property
    def drop_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.role == ColumnRole.IDENTIFIER]


class SchemaAgent:
    """
    Agent that interprets industrial dataset schemas using an LLM.

    When ANTHROPIC_API_KEY is set, uses Claude to interpret column meanings.
    Falls back to a heuristic interpreter when no API key is available,
    making the codebase fully runnable without credentials.

    Example:
        agent = SchemaAgent()
        schema = agent.interpret(
            column_names=["T_react", "P_steam", "F_feed", "purity", "timestamp"],
            documentation="Reactor monitoring data from batch process B-42",
            sample_values={"T_react": [180.2, 181.1], "purity": [98.1, 97.8]},
        )
        print(schema.target_column)     # "purity"
        print(schema.feature_columns)   # ["T_react", "P_steam", "F_feed"]
    """

    SYSTEM_PROMPT = """You are an expert industrial data scientist specialising in
process engineering datasets. Given column names, optional documentation, and
sample values, you interpret each column's physical meaning, units, and role.

Respond ONLY with a JSON object in this exact format:
{
  "domain": "<inferred industry domain>",
  "summary": "<one sentence summary of the dataset>",
  "columns": [
    {
      "name": "<original column name>",
      "semantic_name": "<human-readable interpretation>",
      "role": "target|feature|timestamp|identifier|unknown",
      "units": "<units or null>",
      "description": "<what this column represents>",
      "is_cyclic": false,
      "quality_notes": "<any data quality observations>"
    }
  ]
}"""

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        self.model = model
        self._client = None
        if _HAS_ANTHROPIC:
            try:
                self._client = Anthropic()
            except Exception:
                pass  # No API key — fall back to heuristics

    def interpret(
        self,
        column_names: list[str],
        documentation: str = "",
        sample_values: dict[str, list[Any]] | None = None,
    ) -> SchemaInterpretation:
        """
        Interpret a dataset schema.

        Args:
            column_names:  List of raw column names from the CSV
            documentation: Any accompanying text documentation
            sample_values: Dict mapping column name to a few sample values
        Returns:
            SchemaInterpretation with metadata for each column
        """
        if self._client:
            return self._llm_interpret(column_names, documentation, sample_values or {})
        else:
            return self._heuristic_interpret(column_names, documentation, sample_values or {})

    def _llm_interpret(
        self,
        column_names: list[str],
        documentation: str,
        sample_values: dict[str, list[Any]],
    ) -> SchemaInterpretation:
        """Use Claude to interpret the schema."""
        user_prompt = f"""Dataset columns: {column_names}

Documentation: {documentation or 'None provided'}

Sample values: {json.dumps({k: v[:3] for k, v in sample_values.items()}, default=str)}

Interpret this industrial dataset schema."""

        message = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = message.content[0].text
        return self._parse_llm_response(raw, column_names)

    def _heuristic_interpret(
        self,
        column_names: list[str],
        documentation: str,
        sample_values: dict[str, list[Any]],
    ) -> SchemaInterpretation:
        """Rule-based fallback schema interpreter."""
        timestamp_hints = {"time", "timestamp", "date", "datetime", "t", "ts"}
        target_hints = {"target", "label", "output", "y", "purity", "quality", "yield"}
        id_hints = {"id", "index", "uuid", "batch", "lot"}

        columns: list[ColumnMetadata] = []
        target_col = None
        ts_col = None

        for col in column_names:
            lower = col.lower().strip()
            if any(h in lower for h in timestamp_hints):
                role = ColumnRole.TIMESTAMP
                ts_col = col
            elif any(h in lower for h in target_hints):
                role = ColumnRole.TARGET
                target_col = col
            elif any(h in lower for h in id_hints):
                role = ColumnRole.IDENTIFIER
            else:
                role = ColumnRole.FEATURE

            # Infer units from common industrial abbreviations
            units = None
            if any(x in lower for x in ["temp", "_t", "t_"]):
                units = "°C"
            elif any(x in lower for x in ["press", "_p", "p_"]):
                units = "bar"
            elif any(x in lower for x in ["flow", "_f", "f_"]):
                units = "L/min"
            elif any(x in lower for x in ["pct", "ratio", "purity"]):
                units = "%"

            columns.append(ColumnMetadata(
                name=col,
                semantic_name=col.replace("_", " ").title(),
                role=role,
                units=units,
                description=f"Industrial sensor column: {col}",
            ))

        features = [c.name for c in columns if c.role == ColumnRole.FEATURE]

        return SchemaInterpretation(
            columns=columns,
            domain="industrial process" if documentation else "unknown",
            target_column=target_col,
            timestamp_column=ts_col,
            suggested_features=features,
            summary=f"Dataset with {len(column_names)} columns. "
                    f"Target: {target_col}. Features: {len(features)}.",
        )

    @staticmethod
    def _parse_llm_response(raw: str, column_names: list[str]) -> SchemaInterpretation:
        """Parse Claude's JSON response into SchemaInterpretation."""
        try:
            # Extract JSON from response (may be wrapped in markdown)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            data = json.loads(raw[start:end])
        except (json.JSONDecodeError, ValueError):
            # Fallback: return minimal interpretation
            data = {"domain": "unknown", "summary": raw[:200], "columns": []}

        columns: list[ColumnMetadata] = []
        target_col = None
        ts_col = None

        for col_data in data.get("columns", []):
            role_str = col_data.get("role", "unknown")
            try:
                role = ColumnRole(role_str)
            except ValueError:
                role = ColumnRole.UNKNOWN
            if role == ColumnRole.TARGET:
                target_col = col_data["name"]
            elif role == ColumnRole.TIMESTAMP:
                ts_col = col_data["name"]
            columns.append(ColumnMetadata(
                name=col_data.get("name", ""),
                semantic_name=col_data.get("semantic_name", col_data.get("name", "")),
                role=role,
                units=col_data.get("units"),
                description=col_data.get("description", ""),
                is_cyclic=col_data.get("is_cyclic", False),
                quality_notes=col_data.get("quality_notes", ""),
            ))

        features = [c.name for c in columns if c.role == ColumnRole.FEATURE]
        return SchemaInterpretation(
            columns=columns,
            domain=data.get("domain", "unknown"),
            target_column=target_col,
            timestamp_column=ts_col,
            suggested_features=features,
            summary=data.get("summary", ""),
            raw_llm_response=raw,
        )
