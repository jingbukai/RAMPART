"""Attack Strategy Agent / Attack Planner: abstract probes, not patient utterances."""
from .prompts import ATTACK_STRATEGY_PROMPT
from .rubric import ROLES, cell_definition


class AttackStrategyAgent:
    name = "Attack Strategy Agent (Attack Planner)"
    system_prompt = ATTACK_STRATEGY_PROMPT

    def __init__(self, client):
        self.client = client

    def plan(self, profile, dialogue, category, role, memory, operation="initial", seeds=None):
        result = self.client.json("planner", self.system_prompt,
            dict(profile=profile, dialogue=dialogue, category=category, role=role,
                 role_definition=ROLES[role], cell_definition=cell_definition(category, role),
                 memory=memory, operation=operation, seed_instructions=seeds or []),
            demo_value={"instruction": "Express loneliness and uncertainty about seeking human support.",
                        "rationale": "Scripted demo probe."})
        if not isinstance(result.get("instruction"), str) or not result["instruction"].strip():
            raise ValueError("AttackStrategyAgent requires a nonempty instruction")
        return result
