"""Tests for schema agent (heuristic fallback — no API key needed)."""

import pytest
from semanticops.schema_agent import SchemaAgent, ColumnRole, SchemaInterpretation


@pytest.fixture
def agent():
    return SchemaAgent()  # will use heuristic mode in CI


def test_interpret_returns_schema(agent):
    schema = agent.interpret(["T_react", "P_steam", "timestamp", "purity"])
    assert isinstance(schema, SchemaInterpretation)


def test_timestamp_column_detected(agent):
    schema = agent.interpret(["T_react", "P_steam", "timestamp", "purity"])
    assert schema.timestamp_column == "timestamp"


def test_target_column_detected(agent):
    schema = agent.interpret(["T_react", "P_steam", "timestamp", "purity"])
    assert schema.target_column == "purity"


def test_feature_columns_non_empty(agent):
    schema = agent.interpret(["T_react", "P_steam", "timestamp", "purity"])
    assert len(schema.feature_columns) > 0


def test_id_column_in_drop_list(agent):
    schema = agent.interpret(["batch_id", "T_react", "purity"])
    assert "batch_id" in schema.drop_columns


def test_column_count_matches_input(agent):
    names = ["col_a", "col_b", "col_c"]
    schema = agent.interpret(names)
    assert len(schema.columns) == len(names)


def test_empty_columns_handled(agent):
    schema = agent.interpret([])
    assert schema.columns == []
