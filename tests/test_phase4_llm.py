from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from src.engine.nist_retriever import RetrievalCandidate, RetrievalResult
from src.llm.client import ChatCompletionResult, GroqChatClient, MissingGroqApiKey
from src.llm.config import LLMConfig
from src.llm.narrator import RiskNarrationService, parse_consolidated_risk_narrative
from src.llm.prompts import consolidated_risk_narrative_messages, nist_application_messages, risk_explanation_messages


def test_groq_client_requires_api_key() -> None:
    with pytest.raises(MissingGroqApiKey):
        GroqChatClient(api_key="")


def test_groq_client_parses_chat_completion_response() -> None:
    session = _Session(
        [
            _Response(
                200,
                {
                    "model": "llama-3.3-70b-versatile",
                    "choices": [{"message": {"content": "Grounded answer."}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                },
            )
        ]
    )
    client = GroqChatClient(api_key="test-key", session=session)

    result = client.complete([], max_tokens=50)

    assert result.content == "Grounded answer."
    assert result.usage["prompt_tokens"] == 10
    assert session.calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert session.calls[0]["json"]["model"] == "llama-3.3-70b-versatile"


def test_risk_prompt_contains_evidence_and_anti_hallucination_instruction() -> None:
    row = _risk_row()

    messages = risk_explanation_messages(row)
    combined = "\n".join(message.content for message in messages)

    assert "Use only the evidence provided" in combined
    assert "payment-api-prod-01" in combined
    assert "CVE-SYN-2026-0010" in combined
    assert "Payment Processing" in combined
    assert "Do not invent" in combined


def test_nist_prompt_contains_retrieved_control_prose() -> None:
    row = _risk_row()
    candidate = _candidate()

    messages = nist_application_messages(row, candidate)
    combined = "\n".join(message.content for message in messages)

    assert "SI-2" in combined
    assert "Flaw Remediation" in combined
    assert "Install security-relevant software and firmware updates" in combined
    assert "hybrid_score" in combined


def test_consolidated_risk_prompt_contains_all_required_outputs() -> None:
    messages = consolidated_risk_narrative_messages(_risk_row(), _candidate())
    combined = "\n".join(message.content for message in messages)

    assert "WHY IT MATTERS" in combined
    assert "THREAT SUMMARY" in combined
    assert "NIST APPLICATION" in combined
    assert "payment-api-prod-01" in combined
    assert "SI-2" in combined
    assert "local_threat_ransomware" in combined
    assert "ransomware_signal_source" in combined
    assert "attribute the ransomware signal to CISA KEV" in combined


def test_narration_service_makes_one_call_per_risk() -> None:
    client = _RecordingClient(
        [
            (
                "WHY IT MATTERS: Risk explanation.\n"
                "THREAT SUMMARY: Threat summary.\n"
                "NIST APPLICATION: NIST application."
            ),
        ]
    )
    service = RiskNarrationService(
        client,
        config=LLMConfig(rate_limit_sleep_seconds=0),
    )

    narration = service.narrate_risk(_risk_row(), _retrieval_result())

    assert narration.risk_explanation == "Risk explanation."
    assert narration.nist_application == "NIST application."
    assert narration.threat_summary == "Threat summary."
    assert len(client.calls) == 1


def test_consolidated_parser_requires_all_sections() -> None:
    parsed = parse_consolidated_risk_narrative(
        "WHY IT MATTERS: A\nTHREAT SUMMARY: B\nNIST APPLICATION: C"
    )
    assert parsed == {
        "why_it_matters": "A",
        "threat_summary": "B",
        "nist_application": "C",
    }

    with pytest.raises(ValueError):
        parse_consolidated_risk_narrative("WHY IT MATTERS: A")


def test_phase4_env_constructor_uses_groq_api_key(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    client = GroqChatClient.from_env()
    assert client.api_key == "env-key"


def _risk_row() -> pd.Series:
    return pd.Series(
        {
            "risk_rank": 1,
            "risk_severity": "Critical",
            "composite_risk_score": 0.91,
            "score_drivers": "ransomware association, internet exposure",
            "asset_name": "payment-api-prod-01",
            "asset_type": "API Server",
            "environment": "Production",
            "owner_team": "Payments Team",
            "feature_internet_exposed": 1.0,
            "edr_installed_bool": True,
            "vulnerability_name": "Payment API Insecure Direct Object Reference",
            "cve": "CVE-SYN-2026-0010",
            "cvss": 9.1,
            "affected_component": "Payment Handler",
            "exploit_available_bool": True,
            "patch_available_bool": True,
            "days_open": 11,
            "cveID": "",
            "kev_known_ransomware_bool": False,
            "epss": 0.0,
            "business_service": "Payment Processing",
            "business_owner": "CFO",
            "business_impact": "Payments and fund transfers fail; PCI DSS breach obligations triggered",
            "compliance_scope": "PCI DSS",
            "rto_hours": 1,
            "threat_intel_match_bool": True,
            "campaign_names": "CitrixBleed Exploitation",
            "threat_actors": "IronVeil",
            "threat_summaries": "Targets payment API IDOR vulnerabilities in tandem with CitrixBleed.",
            "threat_ransomware_bool": False,
            "vuln_id": "V-2009",
        }
    )


def _candidate() -> RetrievalCandidate:
    return RetrievalCandidate(
        control_id="SI-2",
        name="Flaw Remediation",
        discussion="Install security-relevant software and firmware updates within organization-defined time periods.",
        semantic_score=0.5,
        bm25_score=1.0,
        control_prior_score=1.0,
        hybrid_score=0.8,
        rank=1,
        prior_reasons=("SI-2: flaw remediation and patching signal",),
    )


def _retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        query="payment API remediation",
        priors=tuple(),
        candidates=(_candidate(),),
    )


@dataclass
class _Response:
    status_code: int
    data: dict
    text: str = ""

    def json(self):
        return self.data


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


class _RecordingClient:
    def __init__(self, contents):
        self.contents = list(contents)
        self.config = LLMConfig(rate_limit_sleep_seconds=0)
        self.calls = []

    def complete(self, messages, *, max_tokens):
        self.calls.append((messages, max_tokens))
        return ChatCompletionResult(
            content=self.contents.pop(0),
            model="llama-3.3-70b-versatile",
            usage={"prompt_tokens": 1, "completion_tokens": 1},
            raw_response={},
        )
