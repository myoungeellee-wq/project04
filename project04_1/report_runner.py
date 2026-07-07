from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from context_builder import ContextBuilder
from rag_config import RagConfig
from sar_template import SarTemplate


@dataclass(frozen=True)
class ReportResult:
    answer: str
    sources: list[dict[str, Any]]
    timings: dict[str, float]
    sar_report: str


class ReportRunner:
    """Coordinates context building, LLM answer generation, and SAR finalization."""

    def __init__(
        self,
        config: RagConfig,
        context_builder: ContextBuilder | None = None,
        sar_template: SarTemplate | None = None,
    ):
        self.config = config
        self.context_builder = context_builder or ContextBuilder(config)
        self.sar_template = sar_template or SarTemplate()

    def build_llm(self) -> ChatOllama:
        kwargs: dict[str, Any] = {
            "model": self.config.ollama_model,
            "base_url": self.config.ollama_base_url,
            "temperature": self.config.ollama_temperature,
        }
        api_key = getattr(self.config, "ollama_api_key", "")
        if api_key:
            kwargs["client_kwargs"] = {
                "headers": {
                    "Authorization": f"Bearer {api_key}",
                }
            }
        return ChatOllama(**kwargs)

    def run(
        self,
        question: str,
        top_k: int = 5,
        auxiliary_models: list[str] | None = None,
    ) -> ReportResult:
        started_at = time.perf_counter()
        bundle = self.context_builder.build(
            question=question,
            top_k=top_k,
            auxiliary_models=auxiliary_models,
        )

        system_prompt = (
            "You are a Korean real-estate RAG assistant for Seoul transaction CSV data. "
            "Answer in Korean. Use only the retrieved context. If the context is insufficient, say so. "
            "Explain amounts in manwon/eokwon when relevant, and include price per square meter when present."
        )
        user_prompt = f"질문:\n{question}\n\n검색 결과:\n{bundle.context_text}\n\n답변:"

        generation_started_at = time.perf_counter()
        response = self.build_llm().invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        generation_seconds = time.perf_counter() - generation_started_at

        timings = {
            "retrieve_seconds": bundle.retrieve_seconds,
            "generation_seconds": generation_seconds,
            "total_seconds": time.perf_counter() - started_at,
        }
        answer = response.content
        sar_report = self.sar_template.assemble(
            question=question,
            answer=answer,
            sources=bundle.contexts,
            timings=timings,
            ollama_model=self.config.ollama_model,
            embedding_model=self.config.embedding_model,
            auxiliary_models=auxiliary_models,
        )
        return ReportResult(
            answer=answer,
            sources=bundle.contexts,
            timings=timings,
            sar_report=sar_report,
        )

    def answer(
        self,
        question: str,
        top_k: int = 5,
        auxiliary_models: list[str] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, float]]:
        result = self.run(question=question, top_k=top_k, auxiliary_models=auxiliary_models)
        return result.answer, result.sources, result.timings
