"""
Unit tests for LLM client parsing and fallback behavior.
"""

from agent.llm_client import LLMClient
from agent.schemas import AgentAction


def test_extract_json_from_clean():
    raw = '{"action": "finish_campaign", "reason": "Target achieved"}'
    parsed = LLMClient.extract_json(raw)
    assert parsed is not None
    assert parsed["action"] == "finish_campaign"


def test_extract_json_with_think_tags():
    raw = (
        "<think>We need to test Random Forest because of non-linearity.</think>\n"
        '```json\n{"action": "run_experiment", "model_family": "RandomForestClassifier", '
        '"hypothesis": "Ensemble reduces variance", "reason": "Better generalization"}\n```'
    )
    parsed = LLMClient.extract_json(raw)
    assert parsed is not None
    assert parsed["action"] == "run_experiment"
    assert parsed["model_family"] == "RandomForestClassifier"


def test_extract_json_with_raw_text_surrounding():
    raw = 'Here is your recommendation: {"action": "compare_experiments", "reason": "Need standings"} Hope this helps!'
    parsed = LLMClient.extract_json(raw)
    assert parsed is not None
    assert parsed["action"] == "compare_experiments"
