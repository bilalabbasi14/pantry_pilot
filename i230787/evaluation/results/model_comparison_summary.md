# Model comparison (2026-10-04T16:29:00+00:00)

Inputs: 10 identical cases per model; fallback disabled; delay 45.0s between cases.

| Model | Task success | Structured validity (valid/decisions) | Correct first action | Semantic rejections | Rate-limited cases | Median latency ms | P90 latency ms | Avg input tok/run | Avg output tok/run | Est. cost for 10 runs (USD) |
|---|---|---|---|---|---|---|---|---|---|---|
| gemini:gemini-3.5-flash-lite | 10/10 | 17/17 | 10/10 | 0 | 0 | 3654 | 5360 | 2988 | 138 | 0.012427 |
| groq:openai/gpt-oss-120b | 10/10 | 18/18 | 10/10 | 1 | 0 | 3032 | 4608 | 3152 | 282 | 0.006843 |
| gemini:gemini-2.5-flash | 4/10 | 8/8 | 3/10 | 1 | 6 | 6175 | 7743 | 3462 | 166 | 0.005812 |

| Case | gemini:gemini-3.5-flash-lite | groq:openai/gpt-oss-120b | gemini:gemini-2.5-flash |
|---|---|---|---|
| M1 Create a grocery plan for spaghetti bolognese for  | PASS completed (first: select_recipes) | PASS completed (first: select_recipes) | PASS completed (first: select_recipes) |
| M2 We're a family of four - put together the shopping | PASS completed (first: select_recipes) | PASS completed (first: select_recipes) | PASS completed (first: select_recipes) |
| M3 Pancakes for 8 and a veggie omelette for 2. I alre | PASS completed (first: select_recipes) | PASS completed (first: select_recipes) | PASS completed (first: build_grocery_plan) |
| M4 Make a grocery plan from the beef tacos recipe. | PASS needs_clarification (first: needs_clarification) | PASS needs_clarification (first: needs_clarification) | FAIL failed (first: None) |
| M5 I need groceries for the salad, 2 servings please. | PASS needs_clarification (first: needs_clarification) | PASS needs_clarification (first: needs_clarification) | FAIL failed (first: None) |
| M6 Plan beef wellington for 4 people. | PASS needs_clarification (first: needs_clarification) | PASS needs_clarification (first: needs_clarification) | FAIL failed (first: None) |
| M7 Turn the attached recipe into a grocery list for 4 | PASS completed (first: import_external_recipe) | PASS completed (first: import_external_recipe) | FAIL failed (first: None) |
| M8 Grocery list for guacamole for 8 please. | PASS completed (first: select_recipes) | PASS completed (first: select_recipes) | FAIL failed (first: None) |
| M9 Buy everything for margherita pizza for 2 on Insta | PASS blocked (first: blocked) | PASS blocked (first: blocked) | PASS blocked (first: blocked) |
| M10 What ingredients are in the vegetable stir-fry? | PASS completed (first: get_recipe) | PASS completed (first: get_recipe) | FAIL failed (first: None) |
