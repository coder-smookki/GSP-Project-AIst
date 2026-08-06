from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr


class OllamaEmbeddingClient:
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self._client = OpenAIEmbeddings(
            model=model,
            base_url=f"{base_url}/v1",
            api_key=SecretStr("sk-no-key"),
            check_embedding_ctx_length=False,
        )
        self._dim: int | None = None

    @property
    def model_name(self) -> str:
        return self._client.model

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("test"))
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return await self._client.aembed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)
