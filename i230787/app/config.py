from pathlib import Path
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All runtime limits live here and can be overridden by environment variables.
    A local .env is read only when present; production uses real environment variables."""
    model_config = SettingsConfigDict(env_file=ROOT / '.env', extra='ignore')
    host: str = '127.0.0.1'
    port: int = 8000
    log_level: str = 'INFO'

    # Default model (selected by evaluation/model_comparison) and optional alternates.
    model_provider: str = 'unconfigured'
    model_name: str = ''
    alternate_models: str = ''  # comma separated provider:model identifiers
    enable_model_fallback: bool = True  # on 429/provider errors try the alternates (other quotas)
    gemini_api_key: SecretStr = SecretStr('')
    groq_api_key: SecretStr = SecretStr('')

    # Agent loop limits.
    max_steps: int = Field(default=6, ge=1, le=6)
    max_tool_retries: int = Field(default=2, ge=0, le=2)
    max_decision_repairs: int = Field(default=2, ge=0, le=2)
    max_output_tokens: int = Field(default=1024, ge=64, le=4096)
    max_run_tokens: int = Field(default=40000, ge=1000)
    run_timeout_seconds: float = Field(default=40, gt=0, le=40)
    model_timeout_seconds: float = Field(default=15, gt=0, le=40)
    min_model_window_seconds: float = Field(default=3, ge=0)
    tool_timeout_seconds: float = Field(default=2, gt=0, le=10)
    max_external_chars: int = Field(default=12000, ge=500)
    max_history_messages: int = Field(default=12, ge=0, le=12)

    # Admission and spend controls for the public deployment.
    max_concurrent_runs: int = Field(default=4, ge=1)
    arena_rate_limit_per_minute: int = Field(default=60, ge=1)
    chat_rate_limit_per_minute: int = Field(default=20, ge=1)
    daily_model_call_limit: int = Field(default=3000, ge=1)


settings = Settings()
