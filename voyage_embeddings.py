"""
LangChain Embeddings implementation backed by the Voyage AI Python client.
"""

from __future__ import annotations

import os
from typing import List

from langchain_core.embeddings import Embeddings
from voyageai import Client as VoyageClient


class VoyageAIEmbeddings(Embeddings):
    """Embeddings via `voyageai.Client.embed` (document vs query input types)."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model or os.getenv("VOYAGE_EMBED_MODEL", "voyage-4")
        self._client = VoyageClient(api_key=api_key or os.getenv("VOYAGE_API_KEY"))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        result = self._client.embed(
            texts,
            model=self._model,
            input_type="document",
        )
        return list(result.embeddings)

    def embed_query(self, text: str) -> List[float]:
        result = self._client.embed(
            [text],
            model=self._model,
            input_type="query",
        )
        return list(result.embeddings[0])
