import pytest

from infrastructure.config.settings import get_settings
from infrastructure.embeddings.ollama_client import OllamaEmbeddingClient


class TestOllamaEmbeddingClient:
    @pytest.fixture
    def client(self):
        settings = get_settings()
        return OllamaEmbeddingClient(
            model=settings.embedding_model,
            base_url=settings.ollama_local_base_url,
        )

    @pytest.mark.embeddings
    def test_embed_returns_vectors(self, client):
        result = client.embed(["hello", "world"])
        assert len(result) == 2
        assert all(len(v) > 0 for v in result)

    @pytest.mark.embeddings
    def test_embed_query_returns_vector(self, client):
        result = client.embed_query("hello")
        assert len(result) > 0

    @pytest.mark.embeddings
    def test_embedding_dim(self, client):
        dim = client.embedding_dim
        assert isinstance(dim, int)
        assert dim > 0
