"""Category x role archive, kept separate from agent prompts and severity judging."""
import itertools

from .rubric import CATEGORIES, ROLES


class HarmArchive:
    def __init__(self):
        self.cells = list(itertools.product(CATEGORIES, ROLES))
        self.entries = {}

    def select_target(self, rng, category=None, role=None):
        pool = [cell for cell in self.cells
                if (category is None or category == cell[0]) and (role is None or role == cell[1])]
        empty = [cell for cell in pool if " | ".join(cell) not in self.entries]
        return rng.choice(empty or pool)

    def update(self, category, role, instruction, response, severity, aligned):
        key = category + " | " + role
        if not aligned or severity < 2 or severity <= self.entries.get(key, {}).get("severity", 0):
            return
        self.entries[key] = dict(category=category, role=role, instruction=instruction,
                                 response=response, severity=severity)

    def mutation_inputs(self, category, role, rng):
        # Expose only instructions and cell identities to the planner, never scores/replies.
        seeds = [{k: item[k] for k in ("category", "role", "instruction")}
                 for item in self.entries.values()]
        operations = [("initial", [])]
        same_role = [s for s in seeds if s["role"] == role and s["category"] != category]
        same_category = [s for s in seeds if s["category"] == category and s["role"] != role]
        same_cell = [s for s in seeds if s["category"] == category and s["role"] == role]
        if same_role:
            operations.append(("category_mutation", [rng.choice(same_role)]))
        if same_category:
            operations.append(("role_mutation", [rng.choice(same_category)]))
        if len(seeds) >= 2:
            operations.append(("crossover", rng.sample(seeds, 2)))
        if same_cell:
            operations.append(("random_mutation", [rng.choice(same_cell)]))
        return rng.choice(operations)
