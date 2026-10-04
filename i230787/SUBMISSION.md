# Submission summary

Copy these details into i230787_submission.pdf. An editable Word version is provided as `i230787_submission.docx`.
URLs below were checked live on 2026-10-04 (all returned HTTP 200; `/health` reported `model_configured: true`).

- Full name: Ahmad Bilal
- Roll number: i230787
- Class / section: C
- University email: i230787@isb.nu.edu.pk
- GitHub username: bilalabbasi14
- Agent name and domain: Pantry Pilot — Recipe & Grocery Planner
- Private GitHub repository URL: https://github.com/bilalabbasi14/pantry_pilot (project files are in the `i230787/` folder)
- Final source commit hash: `d247f8ec8c171cc5e9b4327a9c8287de2cefc189` (evaluated source on `main`; any later commit only updates submission metadata)
- Working public agent interface URL: https://pantry-pilot-i230787.onrender.com/
- GET /health URL: https://pantry-pilot-i230787.onrender.com/health
- POST /arena/run URL: https://pantry-pilot-i230787.onrender.com/arena/run
- GET /arena/manifest URL: https://pantry-pilot-i230787.onrender.com/arena/manifest
- API documentation URL: https://pantry-pilot-i230787.onrender.com/docs
- Hosting provider: Render (free web service, Python runtime, root directory `i230787`)
- Default model and provider: `gemini-3.5-flash-lite` (Google Gemini API), selected by the model-selection experiment
  (`evaluation/model_selection.md`)
- Other available models: `openai/gpt-oss-120b` on Groq (selectable in the UI and automatic fallback on Gemini rate-limit/provider errors)
- Example input and expected behavior:
  - Input: `{"task": "Create a grocery plan for spaghetti bolognese for 6 people and greek salad for 2. I already have olive oil.", "external_context": [], "arena_config": {"max_steps": 6, "fault": "none"}}`
  - Expected: `status: completed`, `stop_reason: goal_completed`, 2 steps (`select_recipes` then `build_grocery_plan`);
    `final_response` lists one merged, categorized grocery list (Produce, Meat & Seafood, Dairy & Eggs, Pantry) with
    quantities scaled to 6 and 2 servings (e.g. ground beef 750 g, spaghetti 600 g) and reports
    "Excluded (already have): olive oil" (olive oil appears in both recipes).
  - Missing information (e.g. "Make a grocery plan from the chicken pasta recipe.") returns `needs_clarification`;
    replying "4" in the same chat continues the same goal.
  - Purchase requests ("order it on Instacart and pay") return `blocked` / `autonomy_boundary`; nothing is bought.
- Cold-start / restart limitations: Render free instances sleep when idle, so the first request after sleep can take
  about 30–60 s while the service starts. Chat history and per-session plans are in memory in a single worker: a restart,
  redeploy or sleep clears them, and a browser reload starts a new chat. Arena runs are stateless, so they are unaffected.
  Model calls depend on free-tier provider quotas: if both providers rate-limit, a run stops safely with `failed` /
  `model_rate_limited`.
- Instructor repository invitation status: [CONFIRM: invitation sent / accepted]
- Public test result file: `evaluation/results/public_results.json` (local, 10/10). Against the deployed URL:
  `evaluation/results/public_results_deployed_paced.json` (10/10, 20 s between cases) and
  `evaluation/results/public_results_deployed.json` (8/10, 3 s between cases: the 2 failures were `model_rate_limited`
  while both free-tier quotas were exhausted, not agent errors). Custom reliability results:
  `evaluation/results/reliability_results.json`; model comparison: `evaluation/results/model_comparison_summary.md`.

Submit i230787.zip and i230787_submission.pdf in Google Classroom, then click Turn in.
The ZIP must contain one i230787/ project folder including this file. See assignment Section 19.
