from .openai_compat import OpenAICompatClient


class LlamaCppClient(OpenAICompatClient):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080",
        timeout: float = 60.0,
    ):
        super().__init__(model=model, base_url=base_url, timeout=timeout)
