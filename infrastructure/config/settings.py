from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    llm_provider: str = "ollama-local"  # ollama-local | llamacpp | vllm
    llm_model: str = "qwen3.6:35b-a3b-nvfp4"
    llm_support_model: str = "None"
    llm_num_ctx: int = 16384
    llm_repeat_penalty: float = 1.1

    ollama_local_base_url: str = "http://localhost:11434"
    llamacpp_base_url: str = "http://localhost:8080"
    vllm_base_url: str = "http://localhost:8000"

    # Embeddings
    embedding_provider: str = "ollama-local"  # ollama-local | llamacpp | vllm
    embedding_model: str = "qwen3-embedding:8b"

    # Observability (OpenLIT / OTLP)
    otlp_endpoint: str = ""  # e.g. http://openlit:4318; empty = tracing disabled

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
