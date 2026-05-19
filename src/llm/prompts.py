from __future__ import annotations

import pandas as pd

from src.engine.nist_retriever import RetrievalCandidate

from .client import ChatMessage
from .evidence import format_evidence_block, nist_evidence, risk_evidence, summarize_top_risks


SYSTEM_MESSAGE = ChatMessage(
    role="system",
    content=(
        "You are a senior cybersecurity analyst writing for a CISO and technical managers. "
        "Use only the evidence provided by the application. Do not invent CVE details, scores, "
        "campaign names, NIST controls, dates, or remediation steps. If evidence is missing, "
        "say it is not available. Be specific, concise, and action-oriented."
    ),
)


def risk_explanation_messages(row: pd.Series) -> list[ChatMessage]:
    evidence = format_evidence_block(risk_evidence(row))
    return [
        SYSTEM_MESSAGE,
        ChatMessage(
            role="user",
            content=(
                "Given the structured evidence below, write 2-3 sentences explaining why this "
                "risk ranks where it does. Do not repeat every number. Focus on the business "
                "and security reasoning. Do not use bullet points.\n\n"
                f"Evidence:\n{evidence}"
            ),
        ),
    ]


def consolidated_risk_narrative_messages(
    row: pd.Series,
    candidate: RetrievalCandidate,
) -> list[ChatMessage]:
    risk = format_evidence_block(risk_evidence(row))
    nist = format_evidence_block(nist_evidence(candidate))
    return [
        SYSTEM_MESSAGE,
        ChatMessage(
            role="user",
            content=(
                "Write a concise three-part narration for this risk using only the evidence below. "
                "Return exactly these labels, each followed by 1-2 sentences: WHY IT MATTERS, "
                "THREAT SUMMARY, NIST APPLICATION. Do not add extra labels or bullets. "
                "Do not invent facts, dates, CVE behavior, campaign names, or NIST recommendations. "
                "Do not call a named threat campaign ransomware-linked unless local_threat_ransomware is Yes. "
                "If ransomware_signal_source is CISA KEV only, attribute the ransomware signal to CISA KEV, not to the campaign. "
                "Do not quote raw internal feature values or long decimal scores; if a score is needed, round it cleanly.\n\n"
                f"Risk evidence:\n{risk}\n\n"
                f"NIST evidence:\n{nist}"
            ),
        ),
    ]


def nist_application_messages(row: pd.Series, candidate: RetrievalCandidate) -> list[ChatMessage]:
    risk = format_evidence_block(risk_evidence(row))
    nist = format_evidence_block(nist_evidence(candidate))
    return [
        SYSTEM_MESSAGE,
        ChatMessage(
            role="user",
            content=(
                "Explain how the retrieved NIST SP 800-53 control applies to this specific risk. "
                "Use the NIST control evidence below, not model memory. Write 2-3 sentences. "
                "Name the control ID and summarize the practical recommendation. Do not use bullets.\n\n"
                f"Risk evidence:\n{risk}\n\n"
                f"NIST evidence:\n{nist}"
            ),
        ),
    ]


def threat_summary_messages(row: pd.Series) -> list[ChatMessage]:
    evidence = format_evidence_block(risk_evidence(row))
    return [
        SYSTEM_MESSAGE,
        ChatMessage(
            role="user",
            content=(
                "Write a one- or two-sentence threat intelligence note for this risk. Use only "
                "the matched threat evidence. If there is no matched threat intelligence, say "
                "there is no matching campaign in the current environment. Do not use bullets.\n\n"
                f"Evidence:\n{evidence}"
            ),
        ),
    ]


def executive_brief_messages(rows: list[pd.Series]) -> list[ChatMessage]:
    summary = summarize_top_risks(rows)
    return [
        SYSTEM_MESSAGE,
        ChatMessage(
            role="user",
            content=(
                "Write a concise executive brief for the top cyber risks. The audience is the CISO "
                "preparing a board update. Use 4-6 sentences, no bullets. Cover the common pattern, "
                "business exposure, and immediate priority. Use only this evidence:\n\n"
                f"{summary}"
            ),
        ),
    ]
