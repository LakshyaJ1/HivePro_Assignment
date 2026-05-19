from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ControlPrior:
    control_id: str
    weight: float
    reason: str


def infer_control_priors(risk: pd.Series | dict[str, Any]) -> list[ControlPrior]:
    text = _risk_text(risk)
    priors: list[ControlPrior] = []

    if _matches(text, r"\b(eol|end of life|end-of-life|unsupported|no patches|windows server 2012|php 7\.4|postgresql 11|ubuntu 18\.04)\b"):
        priors.extend(
            [
                ControlPrior("SA-22", 1.0, "unsupported component signal"),
                ControlPrior("SI-2", 0.60, "patching gap signal"),
                ControlPrior("CM-8", 0.45, "component inventory signal"),
            ]
        )

    if _matches(text, r"\b(teamcity|jenkins|build pipeline|build chain|ci/cd|cicd|devops|developer|source code|secret|secrets)\b"):
        priors.extend(
            [
                ControlPrior("CM-6", 1.0, "build pipeline security configuration signal"),
                ControlPrior("AC-2", 0.85, "build-system account control signal"),
                ControlPrior("IA-5", 0.80, "build secret and authenticator management signal"),
                ControlPrior("SI-2", 0.45, "software flaw remediation support signal"),
            ]
        )

    if _matches(text, r"\b(authentication|auth|account|credential|password|session|token|ntlm|mfa|privilege|permissions?|bypass)\b"):
        priors.extend(
            [
                ControlPrior("AC-2", 1.0, "account and access management signal"),
                ControlPrior("IA-2", 0.90, "authentication bypass signal"),
                ControlPrior("IA-5", 0.80, "credential or token management signal"),
                ControlPrior("SI-2", 0.40, "patching support signal for exploited auth flaw"),
            ]
        )

    if _matches(text, r"\b(edr|endpoint control|missing endpoint|detection|monitoring|telemetry)\b"):
        priors.extend(
            [
                ControlPrior("AC-2", 1.0, "executive endpoint account exposure signal"),
                ControlPrior("SI-4", 0.90, "endpoint monitoring and telemetry signal"),
                ControlPrior("IA-2", 0.65, "credential protection signal"),
            ]
        )

    if _matches(text, r"\b(rce|remote code|code execution|og nl|ognl|injection|heap buffer|buffer overflow|plugin|tomcat|confluence|jira|php|wordpress|firmware|openssl|openssh)\b"):
        weight = 0.75 if _matches(text, r"\b(authentication|auth|session|token|bypass|teamcity|build)\b") else 1.0
        priors.extend(
            [
                ControlPrior("SI-2", weight, "software flaw remediation signal"),
                ControlPrior("RA-5", 0.75, "vulnerability monitoring signal"),
            ]
        )

    if _matches(text, r"\b(container|kubernetes|dashboard|privileged|image|configuration|misconfiguration|firewall|bucket|storage|encryption|tls)\b"):
        priors.extend(
            [
                ControlPrior("CM-6", 0.85, "secure configuration signal"),
                ControlPrior("RA-5", 0.65, "configuration exposure monitoring signal"),
            ]
        )

    if _matches(text, r"\b(ransomware|incident|response|lateral movement|exfiltrat|campaign|threat actor|actively exploiting|active exploitation)\b"):
        incident_weight = 0.95 if _matches(text, r"\b(confluence|jira|rce|remote code|code execution|collaboration breach)\b") else 0.65
        priors.extend(
            [
                ControlPrior("IR-4", incident_weight, "active campaign incident handling signal"),
                ControlPrior("IR-5", 0.60, "incident monitoring signal"),
            ]
        )

    return _deduplicate_priors(priors)


def infer_control_intent_terms(risk: pd.Series | dict[str, Any]) -> str:
    text = _risk_text(risk)
    terms: list[str] = []

    if _matches(text, r"\b(teamcity|jenkins|build pipeline|build chain|ci/cd|cicd|devops|secret|secrets)\b"):
        terms.append(
            "build pipeline configuration settings developer account governance authenticator secrets management"
        )
    if _matches(text, r"\b(authentication|auth|account|credential|password|session|token|ntlm|mfa|privilege|permissions?|bypass)\b"):
        terms.append(
            "account management identification authentication authenticator management session token credential bypass"
        )
    if _matches(text, r"\b(edr|endpoint control|missing endpoint|detection|monitoring|telemetry)\b"):
        terms.append("system monitoring endpoint telemetry account exposure credential protection")
    if _matches(text, r"\b(eol|end of life|end-of-life|unsupported|no patches)\b"):
        terms.append("unsupported system components replacement vendor support security patches")
    if _matches(text, r"\b(rce|remote code|code execution|ognl|injection|heap buffer|buffer overflow|plugin|firmware|php|wordpress)\b"):
        terms.append("flaw remediation vulnerability monitoring security relevant updates software firmware")
    if _matches(text, r"\b(ransomware|incident|response|lateral movement|exfiltrat|campaign|threat actor|active exploitation)\b"):
        terms.append("incident handling incident monitoring active campaign containment eradication recovery")

    return " ".join(dict.fromkeys(term for term in terms if term))


def prior_score_for_control(control_id: str, priors: list[ControlPrior]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    full_control_id = str(control_id).strip().upper()
    base_control_id = _base_control_id(control_id)

    for prior in priors:
        prior_full_id = str(prior.control_id).strip().upper()
        prior_base_id = _base_control_id(prior.control_id)
        if full_control_id == prior_full_id:
            score = max(score, prior.weight)
            reasons.append(f"{prior.control_id}: {prior.reason}")
        elif base_control_id == prior_base_id:
            score = max(score, prior.weight * 0.6)
            reasons.append(f"{prior.control_id} family: {prior.reason}")

    return min(score, 1.0), list(dict.fromkeys(reasons))


def _risk_text(risk: pd.Series | dict[str, Any]) -> str:
    fields = [
        "vulnerability_name",
        "affected_component",
        "asset_type",
        "asset_name",
        "business_service",
        "campaign_names",
        "threat_actors",
        "threat_summaries",
        "score_drivers",
        "cve",
    ]
    values = []
    for field in fields:
        value = risk.get(field, "") if hasattr(risk, "get") else ""
        if value is not None:
            values.append(str(value))
    return " ".join(values).lower()


def _matches(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def _base_control_id(control_id: str) -> str:
    return str(control_id).split("(")[0].strip().upper()


def _deduplicate_priors(priors: list[ControlPrior]) -> list[ControlPrior]:
    by_id: dict[str, ControlPrior] = {}
    for prior in priors:
        existing = by_id.get(prior.control_id)
        if existing is None or prior.weight > existing.weight:
            by_id[prior.control_id] = prior
    return sorted(by_id.values(), key=lambda item: item.weight, reverse=True)
