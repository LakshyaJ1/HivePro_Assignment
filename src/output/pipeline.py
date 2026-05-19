from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.llm.config import LLMConfig
from src.llm.pipeline import run_phase4_pipeline

from .markdown import render_markdown_report, write_markdown_report
from .models import MarkdownReport


@dataclass(frozen=True)
class Phase5PipelineResult:
    report: MarkdownReport


def run_phase5_pipeline(
    *,
    top_n: int = 5,
    output_dir: Path | str = "reports",
    llm_config: LLMConfig | None = None,
) -> Phase5PipelineResult:
    phase4 = run_phase4_pipeline(top_n=top_n, llm_config=llm_config)
    report = render_markdown_report(
        narration=phase4.report,
        risk_retrievals=phase4.risk_retrievals,
    )
    written = write_markdown_report(report, output_dir)
    return Phase5PipelineResult(report=written)

