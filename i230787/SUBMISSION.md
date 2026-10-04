# Submission summary

Copy these details into i230787_submission.pdf. An editable Word version is provided as `i230787_submission.docx`.
Fields marked TO BE FILLED depend on manual GitHub setup and deployment and were deliberately left blank.

- Full name: Ahmad Bilal
- Roll number: i230787
- Class / section: C
- University email: i230787@isb.nu.edu.pk
- GitHub username: [TO BE FILLED AFTER GITHUB SETUP]
- Agent name and domain: Pantry Pilot — Recipe & Grocery Planner
- Private GitHub repository URL: [TO BE FILLED AFTER MANUAL GITHUB SETUP]
- Final source commit hash: [TO BE FILLED AFTER MANUAL GIT COMMIT]
- Working public agent interface URL: [TO BE FILLED AFTER MANUAL DEPLOYMENT]
- GET /health URL: [TO BE FILLED AFTER MANUAL DEPLOYMENT] (path: `/health`)
- POST /arena/run URL: [TO BE FILLED AFTER MANUAL DEPLOYMENT] (path: `/arena/run`)
- GET /arena/manifest URL: [TO BE FILLED AFTER MANUAL DEPLOYMENT] (path: `/arena/manifest`)
- API documentation URL: [TO BE FILLED AFTER MANUAL DEPLOYMENT] (path: `/docs`)
- Hosting provider: Render (planned; free web service) — [TO BE FILLED AFTER MANUAL DEPLOYMENT]
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
- Instructor repository invitation status: [TO BE FILLED AFTER MANUAL GITHUB SETUP]
- Public test result file: `evaluation/results/public_results.json` (custom reliability results:
  `evaluation/results/reliability_results.json`; model comparison: `evaluation/results/model_comparison_summary.md`)

Submit i230787.zip and i230787_submission.pdf in Google Classroom, then click Turn in.
The ZIP must contain one i230787/ project folder including this file. See assignment Section 19.
