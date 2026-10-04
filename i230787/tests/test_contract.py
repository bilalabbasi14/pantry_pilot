"""Structured-output parsing, schema validation, semantic validation and the autonomy detector."""
import os
import sys
import unittest
sys.path.insert(0, os.path.dirname(__file__))
import offline  # noqa: E402
from app.agent import parse_decision  # noqa: E402
from app.grocery import load_catalog  # noqa: E402
from app.models import AgentDecision, AgentState, PlanSandbox  # noqa: E402
from app.policy import GroundingText, detect_boundary_requests, leaks_system_prompt, validate_semantics  # noqa: E402
from app.prompts import CANARY, SYSTEM_PROMPT  # noqa: E402


class ParseTests(unittest.TestCase):
    def test_valid_decision_and_code_fence(self):
        d, err, _ = parse_decision('```json\n' + offline.select(('pancakes', 4)) + '\n```')
        self.assertIsNone(err)
        self.assertEqual(d.typed_arguments().items[0].servings, 4)

    def test_rejections(self):
        cases = {
            'not json at all': 'not a single valid JSON',
            '[1, 2]': 'not a single valid JSON',
            offline.decision('teleport'): 'status',
            offline.decision('continue', 'run_shell', {'cmd': 'rm -rf /'}): 'action',
            offline.decision('continue', None): 'requires an action',
            offline.decision('continue', 'select_recipes', {'items': [{'recipe_id': 'pancakes', 'servings': 'four'}]}): 'servings',
            offline.decision('continue', 'select_recipes', {'items': [{'recipe_id': 'pancakes', 'servings': 500}]}): 'servings',
            offline.decision('continue', 'get_recipe', {'recipe_id': 'pancakes', 'extra': 1}): 'extra',
            offline.decision('needs_clarification', None): 'user_message',
            offline.decision('blocked', 'get_recipe', {'recipe_id': 'pancakes'}, 'no'): 'must not include an action',
            offline.decision('completed', 'select_recipes', {'items': [{'recipe_id': 'pancakes', 'servings': 2}]}): 'completed',
        }
        for raw, expected in cases.items():
            d, err, _ = parse_decision(raw)
            self.assertIsNone(d, raw)
            self.assertIn(expected, err, raw)

    def test_unknown_top_level_fields_are_dropped_with_warning(self):
        d, err, warnings = parse_decision('{"status":"needs_clarification","user_message":"How many?","thoughts":"x"}')
        self.assertIsNone(err)
        self.assertIn('thoughts', warnings[0])


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.recipes = load_catalog()

    def check(self, raw, task, sandbox=None, tools_used=(), selection_changed=False):
        decision = AgentDecision.model_validate_json(raw)
        state = AgentState(request_id='r', user_goal=task, model='m', max_steps=6, sandbox=sandbox or PlanSandbox(),
                           tools_used=list(tools_used), selection_changed=selection_changed)
        return validate_semantics(decision, state, self.recipes, 0, GroundingText(task))

    def test_grounded_selection_passes(self):
        self.assertEqual(self.check(offline.select(('beef-tacos', 6)), 'tacos for six people please'), [])
        self.assertEqual(self.check(offline.select(('spaghetti-bolognese', 8)), 'double the spag bol'), [])

    def test_all_recipes_request_hits_plan_size_limit(self):
        ids = list(self.recipes)
        planned = PlanSandbox(selected={r: 4 for r in ids[:4]})
        errors = self.check(offline.select(*[(r, 4) for r in ids[4:9]]), 'plan every recipe you have for 4', planned)
        self.assertFalse(any('never asked' in e for e in errors))
        self.assertTrue(any('at most 8 recipes' in e for e in errors))

    def test_invented_servings_rejected(self):
        errors = self.check(offline.select(('beef-tacos', 4)), 'make me a taco grocery list')
        self.assertTrue(any('not stated by the user' in e for e in errors))

    def test_unrequested_recipe_rejected(self):
        errors = self.check(offline.select(('lentil-soup', 4)), 'tacos for 4')
        self.assertTrue(any('never asked' in e for e in errors))

    def test_unknown_recipe_rejected(self):
        errors = self.check(offline.select(('lasagna', 4)), 'lasagna for 4')
        self.assertTrue(any('does not exist' in e for e in errors))

    def test_recipe_default_requires_user_request(self):
        raw = offline.decision('continue', 'select_recipes', {'items': [{'recipe_id': 'pancakes', 'servings': 4, 'servings_source': 'recipe_default'}]})
        self.assertTrue(self.check(raw, 'pancakes please'))
        self.assertEqual(self.check(raw, 'pancakes, keep the recipe default servings'), [])

    def test_remove_must_target_planned_recipe(self):
        raw = offline.decision('continue', 'remove_recipes', {'recipe_ids': ['beef-tacos']})
        self.assertTrue(self.check(raw, 'remove the tacos'))
        self.assertEqual(self.check(raw, 'remove the tacos', PlanSandbox(selected={'beef-tacos': 4})), [])

    def test_exclusions_must_come_from_user(self):
        sandbox = PlanSandbox(selected={'pancakes': 4})
        self.assertEqual(self.check(offline.build(['milk']), 'pancakes for 4, I have milk', sandbox), [])
        self.assertTrue(self.check(offline.build(['milk', 'eggs']), 'pancakes for 4, I have milk', sandbox))

    def test_build_requires_selection(self):
        self.assertTrue(any('no recipes' in e for e in self.check(offline.build(), 'make a plan')))

    def test_completion_requires_satisfied_goal(self):
        done = offline.decision('completed', user_message='done')
        self.assertTrue(self.check(done, 'tacos for 4'))  # nothing done yet
        stale = PlanSandbox(selected={'beef-tacos': 4}, plan_stale=True)
        self.assertTrue(self.check(done, 'tacos for 4', stale, ['select_recipes'], True))


class BoundaryAndLeakTests(unittest.TestCase):
    def test_consequential_requests_detected(self):
        self.assertIn('external_order_or_delivery', detect_boundary_requests('Order the tacos ingredients on Instacart'))
        self.assertIn('purchase_or_payment', detect_boundary_requests('buy them for me and pay with my credit card'))
        self.assertIn('external_communication', detect_boundary_requests('email the list to my wife'))

    def test_ordinary_planning_language_not_flagged(self):
        for text in ['Make a list of what I need to buy for pancakes for 4',
                     "Don't order anything, just plan tacos for 2",
                     'What do I need to purchase for the curry for 6?']:
            self.assertEqual(detect_boundary_requests(text), [], text)

    def test_leak_guard(self):
        self.assertTrue(leaks_system_prompt(f'my marker is {CANARY}', SYSTEM_PROMPT, CANARY))
        self.assertTrue(leaks_system_prompt(SYSTEM_PROMPT[1200:2200], SYSTEM_PROMPT, CANARY))
        self.assertFalse(leaks_system_prompt('Here is your grocery plan for tacos.', SYSTEM_PROMPT, CANARY))


if __name__ == '__main__':
    unittest.main()
