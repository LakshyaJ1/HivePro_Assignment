from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from src.engine.nist_retriever import RetrievalResult

from .client import ChatCompletionResult, GroqChatClient
from .config import LLMConfig
from .prompts import (
    consolidated_risk_narrative_messages,
    executive_brief_messages,
)


@dataclass(frozen=True)
class RiskNarration:
    risk_rank: int
    vuln_id: str
    risk_explanation: str
    nist_application: str
    threat_summary: str
    model: str
    usage: dict[str, int | float | str]


@dataclass(frozen=True)
class LLMReportNarration:
    executive_brief: str
    risks: tuple[RiskNarration, ...]
    model: str
    total_usage: dict[str, int]


class RiskNarrationService:
    """Evidence-first LLM narrator for scored risks and retrieved NIST controls."""

    def __init__(self, client: GroqChatClient, config: LLMConfig | None = None) -> None:
        self.client = client
        self.config = config or client.config

    def narrate_risk(self, row: pd.Series, retrieval: RetrievalResult) -> RiskNarration:
        narrative = self._complete(
            consolidated_risk_narrative_messages(row, retrieval.best),
            max_tokens=self.config.risk_narrative_max_tokens,
        )
        sections = parse_consolidated_risk_narrative(narrative.content)

        return RiskNarration(
            risk_rank=int(row.get("risk_rank", 0)),
            vuln_id=str(row.get("vuln_id", "")),
            risk_explanation=sections["why_it_matters"],
            nist_application=sections["nist_application"],
            threat_summary=sections["threat_summary"],
            model=narrative.model,
            usage=_merge_usage(narrative.usage),
        )

    def narrate_report(
        self,
        risk_retrievals: list[tuple[pd.Series, RetrievalResult]],
    ) -> LLMReportNarration:
        risk_narrations: list[RiskNarration] = []
        for row, retrieval in risk_retrievals:
            risk_narrations.append(self.narrate_risk(row, retrieval))
            self._sleep_for_rate_limit()

        executive = self._complete(
            executive_brief_messages([row for row, _ in risk_retrievals]),
            max_tokens=self.config.executive_brief_max_tokens,
        )

        total_usage = _merge_usage(
            *(risk.usage for risk in risk_narrations),
            executive.usage,
        )
        return LLMReportNarration(
            executive_brief=executive.content,
            risks=tuple(risk_narrations),
            model=executive.model,
            total_usage=total_usage,
        )

    def _complete(self, messages, *, max_tokens: int) -> ChatCompletionResult:
        return self.client.complete(messages, max_tokens=max_tokens)

    def _sleep_for_rate_limit(self) -> None:
        if self.config.rate_limit_sleep_seconds > 0:
            time.sleep(self.config.rate_limit_sleep_seconds)


def _merge_usage(*usages: dict) -> dict[str, int]:
    merged: dict[str, int] = {}
    for usage in usages:
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + int(value)
    return merged


def parse_consolidated_risk_narrative(content: str) -> dict[str, str]:
    labels = {
        "WHY IT MATTERS": "why_it_matters",
        "THREAT SUMMARY": "threat_summary",
        "NIST APPLICATION": "nist_application",
    }
    parsed = {value: "" for value in labels.values()}
    current_key: str | None = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched_label = None
        for label, key in labels.items():
            if line.upper().startswith(label):
                matched_label = (label, key)
                break
        if matched_label:
            label, key = matched_label
            current_key = key
            remainder = line[len(label) :].lstrip(" :-")
            if remainder:
                parsed[key] = _append_sentence(parsed[key], remainder)
            continue
        if current_key:
            parsed[current_key] = _append_sentence(parsed[current_key], line)

    if not all(parsed.values()):
        raise ValueError(
            "Consolidated risk narrative did not contain all required sections: "
            "WHY IT MATTERS, THREAT SUMMARY, NIST APPLICATION"
        )
    return parsed


def _append_sentence(existing: str, value: str) -> str:
    return f"{existing} {value}".strip() if existing else value.strip()
