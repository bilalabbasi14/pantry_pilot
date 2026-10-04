"""Test helpers. Importing this module first forces offline settings, so no test can reach a
paid provider even when a developer's .env contains real keys."""
import json
import os

os.environ.update(GEMINI_API_KEY='', GROQ_API_KEY='', MODEL_PROVIDER='unconfigured', MODEL_NAME='',
                  ALTERNATE_MODELS='', LOG_LEVEL='WARNING')

from app.llm import LLMError, LLMResult  # noqa: E402  (must follow the environment override)


def decision(status='continue', action=None, arguments=None, user_message=None, **extra) -> str:
    body = {'status': status, 'action': action, 'arguments': arguments or {}, 'user_message': user_message,
            'reason': 'scripted', **extra}
    return json.dumps(body)


def select(*items, status='continue') -> str:
    return decision(status, 'select_recipes', {'items': [{'recipe_id': r, 'servings': n} for r, n in items]})


def build(exclude=(), status='completed', message='Plan ready.') -> str:
    return decision(status, 'build_grocery_plan', {'exclude_items': list(exclude)},
                    message if status == 'completed' else None)


class ScriptedClient:
    """Plays back model outputs (str, LLMError, or callable(messages) -> str) and records prompts."""
    def __init__(self, *outputs, input_tokens=120, output_tokens=30):
        self.outputs, self.prompts = list(outputs), []
        self.input_tokens, self.output_tokens = input_tokens, output_tokens

    async def complete(self, messages, max_output_tokens, timeout):
        self.prompts.append(messages)
        if not self.outputs:
            raise AssertionError('scripted model called more times than expected')
        out = self.outputs.pop(0)
        if isinstance(out, LLMError):
            raise out
        text = out(messages) if callable(out) else out
        return LLMResult(text, self.input_tokens, self.output_tokens, 5.0)

    def prompt_text(self, call: int = -1) -> str:
        return '\n'.join(str(m.content) for m in self.prompts[call])
