import os
from enum import Enum
from typing import Literal, Optional, Any, Dict, Iterable

# Try to import RunnableConfig; fall back to a simple alias to avoid import-time failures
try:
    from langchain_core.runnables import RunnableConfig  # type: ignore
except Exception:
    RunnableConfig = Dict[str, Any]  # type: ignore

# Try to import Pydantic; fall back to lightweight shims so dev tooling won't crash
try:
    from pydantic import Field, BaseModel  # type: ignore
    _HAS_PYDANTIC = True
except Exception:
    _HAS_PYDANTIC = False

    def Field(default=None, title: Optional[str] = None, description: Optional[str] = None):
        # Minimal shim: return the default value directly
        return default

    class BaseModel:  # type: ignore
        # Minimal shim to accept kwargs and store as attributes
        def __init__(self, **data: Any):
            for k, v in data.items():
                setattr(self, k, v)

        # Provide a pydantic-like attribute presence for compatibility
        model_fields: Dict[str, Any] = {}


class SearchAPI(Enum):
    DUCKDUCKGO = "duckduckgo"


def _field_names(cls) -> Iterable[str]:
    """
    Get field names in a way that works whether Pydantic is available or not.
    """
    if hasattr(cls, "model_fields") and isinstance(getattr(cls, "model_fields"), dict) and cls.model_fields:
        return cls.model_fields.keys()  # pydantic v2 path
    # Fallback to explicitly listing known fields
    return (
        "max_web_research_loops",
        "local_llm",
        "llm_provider",
        "search_api",
        "fetch_full_page",
        "base_url",
        "strip_thinking_tokens",
    )


class Configuration(BaseModel):
    """The configurable fields for the research assistant."""

    max_web_research_loops: int = Field(
        default=3,
        title="Research Depth",
        description="Number of research iterations to perform",
    )
    local_llm: str = Field(
        # Remove trailing space and use a sensible default name
        default="qwen/qwen3-8b",
        title="LLM Model Name",
        description="Name of the LLM model to use",
    )
    llm_provider: Literal["openai"] = Field(
        default="openai",
        title="LLM Provider",
        description="Provider for the LLM",
    )
    search_api: Literal["duckduckgo"] = Field(
        default="duckduckgo",
        title="Search API",
        description="Web search API to use",
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results",
    )
    base_url: str = Field(
        default="http://127.0.0.1:1234/v1",
        title="Base URL",
        description="Base URL for API",
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses",
    )

    # When Pydantic is not available, expose a compatible model_fields mapping
    if not _HAS_PYDANTIC:
        model_fields = {  # type: ignore[assignment]
            name: None
            for name in (
                "max_web_research_loops",
                "local_llm",
                "llm_provider",
                "search_api",
                "fetch_full_page",
                "base_url",
                "strip_thinking_tokens",
            )
        }

    @classmethod
    def _collect_raw_values(cls, source: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Collect values from environment and a provided source (configurable/context).
        Environment variables take precedence if set. Keys are derived from field names and upper-cased.
        """
        source = source or {}
        values: Dict[str, Any] = {}
        for name in _field_names(cls):
            env_key = name.upper()
            env_val = os.environ.get(env_key)
            val = env_val if env_val is not None else source.get(name)
            if val is not None:
                values[name] = val
        return values

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """
        Create a Configuration instance from a RunnableConfig (LangChain).
        """
        configurable: Dict[str, Any] = {}
        if isinstance(config, dict):
            configurable = config.get("configurable", {}) or {}
        raw_values = cls._collect_raw_values(configurable)
        return cls(**raw_values)

    @classmethod
    def from_context(cls, context: Optional[dict] = None) -> "Configuration":
        """
        Create a Configuration instance from a LangGraph context object defined by context_schema.
        Supports dict input or object with attributes.
        """
        if context is None:
            return cls()

        source: Dict[str, Any] = {}
        if isinstance(context, dict):
            source = context
        else:
            for name in _field_names(cls):
                if hasattr(context, name):
                    source[name] = getattr(context, name)

        raw_values = cls._collect_raw_values(source)
        return cls(**raw_values)