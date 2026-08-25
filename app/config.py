"""Runtime configuration.

Every value is read from the environment so the Docker image can ship
without secrets baked into it.
"""

from datetime import date
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

# .env deliberately overrides inherited shell variables. A stale GROQ_API_KEY
# exported into a developer's shell silently shadows the key they just edited
# here, and the failure looks like a dead account rather than a config bug.
# Docker never ships a .env (.dockerignore excludes it), so containers still
# read real environment variables and nothing is baked into the image.
load_dotenv(REPO_ROOT / ".env", override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Two tiers: a small model decides where a message goes, the large one
    # only runs when a message actually needs tool-calling reasoning.
    # The brief named llama-3.3-70b-versatile / llama-3.1-8b-instant, but this
    # Groq account has neither (404, not 401). These are the closest available
    # equivalents, both verified for JSON mode and native tool_calls.
    router_model: str = "openai/gpt-oss-20b"
    agent_model: str = "openai/gpt-oss-120b"

    orders_path: Path = REPO_ROOT / "data" / "orders.json"
    policy_path: Path = REPO_ROOT / "data" / "trendly_policy.md"

    # orders.json is a fixed fixture set whose dates were written relative to
    # a specific "today" -- 2026-07-29. Reading the wall clock instead makes
    # the fixtures self-contradictory and rots the tests daily. See SOLUTION.md.
    reference_date: date = date(2026, 7, 29)

    max_tool_iterations: int = 6
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()


def today() -> date:
    """The only clock the app reads, so date logic stays deterministic."""
    return get_settings().reference_date
