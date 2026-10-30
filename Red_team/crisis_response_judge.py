"""Conditional crisis diagnostic, independent of target-cell ASR."""
from common import boolean_field, score_field, text_field
from .prompts import CRISIS_JUDGE_PROMPT
from .rubric import CRISIS_DUTIES, CRISIS_RISK, CRISIS_STATUSES


class CrisisResponseJudge:
    name = "Crisis Response Judge"
    system_prompt = CRISIS_JUDGE_PROMPT

    def __init__(self, client):
        self.client = client

    def evaluate(self, prompt_messages, response):
        result = self.client.json("crisis_judge", self.system_prompt,
            dict(prompt_messages=prompt_messages, current_response=response,
                 supported_risk_levels=CRISIS_RISK, response_duties=CRISIS_DUTIES),
            demo_value=dict(applicable=False, risk_level="none", harm_severity=1,
                dimensions={name: "not_required" for name in CRISIS_DUTIES},
                reasoning="Scripted demo diagnostic; no crisis is disclosed."))
        boolean_field(result, "applicable")
        score_field(result, "harm_severity")
        text_field(result, "reasoning")
        if result.get("risk_level") not in CRISIS_RISK:
            raise ValueError("Unknown crisis risk level")
        dimensions = result.get("dimensions")
        if not isinstance(dimensions, dict) or set(dimensions) != set(CRISIS_DUTIES) or any(
                value not in CRISIS_STATUSES for value in dimensions.values()):
            raise ValueError("CrisisResponseJudge requires all five duty statuses")
        if result["applicable"] != (result["risk_level"] != "none"):
            raise ValueError("Crisis applicability and risk level disagree")
        return result
