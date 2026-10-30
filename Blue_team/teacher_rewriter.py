"""Teacher Rewriter: repair using visible dialogue without attack metadata."""
from common import text_field, visible_context
from .policy import SAFE_DEMO
from .prompts import TEACHER_REWRITER_PROMPT


class TeacherRewriter:
    name = "Teacher Rewriter"
    system_prompt = TEACHER_REWRITER_PROMPT

    def __init__(self, client):
        self.client = client

    def rewrite(self, prompt_messages, rejected_text):
        prior, current = visible_context(prompt_messages)
        result = self.client.json("teacher", self.system_prompt,
            dict(prior_dialogue=prior, current_patient_message=current,
                 proposed_response_to_replace=rejected_text),
            demo_value={"response": SAFE_DEMO})
        return text_field(result, "response")
