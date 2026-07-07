from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reporting import build_sar_report


@dataclass(frozen=True)
class SarTemplate:
    """Assembles the final SAR markdown report."""

    def assemble(
        self,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        timings: dict[str, float] | None = None,
        ollama_model: str = "",
        embedding_model: str = "",
        auxiliary_models: list[str] | None = None,
    ) -> str:
        return build_sar_report(
            question=question,
            answer=answer,
            sources=sources,
            timings=timings,
            ollama_model=ollama_model,
            embedding_model=embedding_model,
            auxiliary_models=auxiliary_models,
        )
