"""HTTP contracts (unchanged from the starter) followed by the grocery-planner decision/state models."""
from typing import Annotated, Any, Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid')
class ExternalContext(Contract):
    source: str = Field(min_length=1, max_length=100)
    content: str = Field(max_length=10000)
    trust: Literal['untrusted'] = 'untrusted'
class Fault(Contract):
    type: Literal['none', 'tool_timeout', 'malformed_tool_output', 'invalid_agent_decision'] = 'none'
    trigger: Literal['first_matching_operation'] = 'first_matching_operation'
class ArenaConfig(Contract):
    max_steps: int = Field(default=6, ge=1, le=6)
    fault: Fault = Field(default_factory=Fault)
    @field_validator('fault', mode='before')
    @classmethod
    def accept_assignment_shorthand(cls, value):
        return {'type': value} if isinstance(value, str) else value
class ArenaRequest(Contract):
    arena_version: Literal['0.1'] = '0.1'
    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=80)
    task: str = Field(min_length=1, max_length=10000)
    external_context: list[ExternalContext] = Field(default_factory=list, max_length=20)
    arena_config: ArenaConfig = Field(default_factory=ArenaConfig)
    @field_validator('task')
    @classmethod
    def nonblank(cls, value):
        if not value.strip(): raise ValueError('Task must not be blank')
        return value
class ToolTrace(Contract):
    step: int = Field(ge=1, le=6)
    tool: str
    attempt: int = Field(default=1, ge=1, le=3)
    outcome: Literal['success', 'timeout', 'malformed_output', 'rejected', 'exception']
    latency_ms: float = Field(default=0, ge=0)
class Metrics(Contract):
    latency_ms: float = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0, le=6)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
class ArenaResponse(Contract):
    arena_version: Literal['0.1'] = '0.1'
    request_id: str
    status: Literal['completed', 'needs_clarification', 'blocked', 'approval_required', 'tool_error', 'contract_error', 'budget_exceeded', 'failed']
    final_response: str = Field(min_length=1, max_length=2000)
    steps: int = Field(default=0, ge=0, le=6)
    stop_reason: str
    tool_calls: list[ToolTrace] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)
class ChatRequest(ArenaRequest):
    session_id: str = Field(min_length=16, max_length=80)
    model: str = Field(default='unconfigured', max_length=120)


# ---------------------------------------------------------------------------
# Grocery-planner domain models
# ---------------------------------------------------------------------------
RecipeId = Annotated[str, Field(pattern=r'^[a-z0-9][a-z0-9-]{0,59}$')]
Category = Literal['Produce', 'Meat & Seafood', 'Dairy & Eggs', 'Bakery', 'Pantry', 'Frozen', 'Other']


class Ingredient(Contract):
    name: str = Field(min_length=1, max_length=60)
    quantity: float | None = Field(default=None, gt=0, le=100000)
    unit: str | None = Field(default=None, max_length=12)
    category: Category = 'Other'


class Recipe(Contract):
    id: RecipeId
    name: str = Field(min_length=1, max_length=80)
    aliases: list[str] = Field(default_factory=list)
    base_servings: int = Field(ge=1, le=50)
    ingredients: list[Ingredient] = Field(min_length=1, max_length=60)
    notes: str = Field(default='', max_length=2000)
    source: Literal['catalog', 'external'] = 'catalog'


class GroceryItem(Contract):
    name: str
    quantity: float | None = Field(default=None, gt=0)
    unit: str | None = None
    category: Category
    recipes: list[str] = Field(min_length=1)


class PlannedRecipe(Contract):
    recipe_id: RecipeId
    name: str
    servings: int = Field(ge=1, le=50)


class GroceryPlan(Contract):
    recipes: list[PlannedRecipe] = Field(min_length=1)
    items: list[GroceryItem] = Field(min_length=1)
    excluded_items: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --- Agent decision contract (what the model is allowed to propose) ----------
ActionName = Literal['select_recipes', 'remove_recipes', 'get_recipe', 'import_external_recipe', 'build_grocery_plan']
DecisionStatus = Literal['continue', 'needs_clarification', 'completed', 'blocked', 'failed']
BoundaryFlag = Literal['purchase_or_payment', 'external_order_or_delivery', 'external_communication',
                       'account_change', 'out_of_domain']


class RecipeServing(Contract):
    recipe_id: RecipeId
    servings: int = Field(ge=1, le=50)
    servings_source: Literal['user', 'recipe_default'] = 'user'


class SelectRecipesArgs(Contract):
    items: list[RecipeServing] = Field(min_length=1, max_length=8)


class RemoveRecipesArgs(Contract):
    recipe_ids: list[RecipeId] = Field(min_length=1, max_length=8)


class GetRecipeArgs(Contract):
    recipe_id: RecipeId


class ImportExternalRecipeArgs(Contract):
    source_index: int = Field(ge=0, lt=20)


class BuildGroceryPlanArgs(Contract):
    exclude_items: list[Annotated[str, Field(min_length=1, max_length=40)]] = Field(default_factory=list, max_length=20)


ACTION_ARGUMENTS: dict[str, type[Contract]] = {
    'select_recipes': SelectRecipesArgs,
    'remove_recipes': RemoveRecipesArgs,
    'get_recipe': GetRecipeArgs,
    'import_external_recipe': ImportExternalRecipeArgs,
    'build_grocery_plan': BuildGroceryPlanArgs,
}


class AgentDecision(Contract):
    """One structured decision. The model proposes; the application validates and executes."""
    status: DecisionStatus
    action: ActionName | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    user_message: str | None = Field(default=None, max_length=800)
    reason: str = Field(default='', max_length=300)
    boundary_flag: BoundaryFlag | None = None

    @model_validator(mode='after')
    def check_shape(self):
        if self.status == 'continue' and self.action is None:
            raise ValueError("status 'continue' requires an action")
        if self.status in ('needs_clarification', 'blocked', 'failed') and self.action is not None:
            raise ValueError(f"status '{self.status}' must not include an action")
        if self.status == 'completed' and self.action not in (None, 'build_grocery_plan'):
            raise ValueError("status 'completed' may only be combined with action 'build_grocery_plan'")
        if self.status in ('needs_clarification', 'blocked') and not (self.user_message or '').strip():
            raise ValueError(f"status '{self.status}' requires a user_message")
        if self.action is None and self.arguments:
            raise ValueError('arguments given without an action')
        if self.action is not None:
            ACTION_ARGUMENTS[self.action].model_validate(self.arguments)
        return self

    def typed_arguments(self) -> Contract | None:
        return None if self.action is None else ACTION_ARGUMENTS[self.action].model_validate(self.arguments)


# --- Tool observations (validated before they may change state) --------------
class SelectRecipesResult(Contract):
    planned_recipes: list[PlannedRecipe]


class RemoveRecipesResult(Contract):
    removed: list[str]
    planned_recipes: list[PlannedRecipe]


class RecipeDetailsResult(Contract):
    recipe: Recipe


class ImportExternalRecipeResult(Contract):
    recipe_id: RecipeId
    name: str
    base_servings: int | None
    ingredient_count: int = Field(ge=1)
    ignored_lines: int = Field(ge=0)
    instruction_like_lines: int = Field(ge=0)


class BuildGroceryPlanResult(Contract):
    plan: GroceryPlan


# --- Sandbox and run state ---------------------------------------------------
class PlanSandbox(Contract):
    """Non-destructive planning workspace. Chat sessions keep it in memory between turns;
    Arena runs always start from an empty one."""
    selected: dict[str, int] = Field(default_factory=dict)
    external_recipes: dict[str, Recipe] = Field(default_factory=dict)
    plan: GroceryPlan | None = None
    plan_stale: bool = False
    pending_goal: str | None = Field(default=None, max_length=10000)
    pending_question: str | None = Field(default=None, max_length=800)


class StepRecord(Contract):
    step: int
    status: str
    action: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    outcome: str
    observation: Any = None


class AgentState(Contract):
    request_id: str
    session_id: str | None = None
    user_goal: str
    model: str
    current_status: str = 'running'
    current_step: int = 0
    max_steps: int
    sandbox: PlanSandbox = Field(default_factory=PlanSandbox)
    selection_changed: bool = False
    tools_used: list[str] = Field(default_factory=list)
    last_decision: dict[str, Any] | None = None
    last_tool_result: Any = None
    step_log: list[StepRecord] = Field(default_factory=list)
    boundary_flags: list[str] = Field(default_factory=list)
    repairs_used: int = 0
    repair_feedback: str | None = None
    model_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    stop_reason: str | None = None
    errors: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
