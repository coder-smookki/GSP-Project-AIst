from .openai_compat import OpenAICompatClient


class OllamaClient(OpenAICompatClient):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        num_ctx: int = 8192,
        repeat_penalty: float = 1.1,
    ):
        super().__init__(
            model=model,
            base_url=base_url,
            timeout=timeout,
            num_ctx=num_ctx,
            repeat_penalty=repeat_penalty,
        )