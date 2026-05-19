from __future__ import annotations

import math

import pandas as pd
from streamlit.testing.v1 import AppTest

import app


def test_app_score_component_frame_has_expected_components() -> None:
    row = pd.Series(
        {
            "feature_cvss": 0.9,
            "contribution_cvss": 0.18,
            "feature_active_exploitation": 1.0,
            "contribution_active_exploitation": 0.15,
            "feature_ransomware": 1.0,
            "contribution_ransomware": 0.2,
            "feature_epss": 0.5,
            "contribution_epss": 0.075,
            "feature_internet_exposed": 1.0,
            "contribution_internet_exposed": 0.1,
            "feature_business_impact": 0.8,
            "contribution_business_impact": 0.08,
            "feature_threat_intel_match": 1.0,
            "contribution_threat_intel_match": 0.05,
            "feature_days_open": 0.3,
            "contribution_days_open": 0.009,
            "feature_missing_edr": 0.0,
            "contribution_missing_edr": 0.0,
        }
    )

    frame = app._score_component_frame(row)

    assert len(frame) == 9
    assert frame["Component"].iloc[0] == "CVSS severity"
    assert frame["Contribution"].sum() > 0.8


def test_app_kev_line_handles_missing_and_confirmed_values() -> None:
    missing = pd.Series({"cveID": float("nan"), "kev_known_ransomware_bool": False})
    confirmed = pd.Series({"cveID": "CVE-2023-4966", "kev_known_ransomware_bool": True})

    assert app._kev_line(missing) == "Not found; ransomware Unknown"
    assert app._kev_line(confirmed) == "Confirmed; ransomware Known"


def test_app_threat_line_handles_nan_match() -> None:
    row = pd.Series({"threat_intel_match_bool": math.nan})

    assert app._threat_line(row) == "No current-environment match"


def test_app_initial_render_is_information_only() -> None:
    at = AppTest.from_file("app.py")
    at.run(timeout=30)

    assert len(at.exception) == 0
    assert at.title[0].value == "TawasolPay Cyber Risk Assistant"
    assert any("No data has been processed yet" in item.value for item in at.info)
    assert len(at.metric) == 0


def test_exception_details_include_type_message_and_traceback() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        details = app._exception_details(exc)

    assert details["type"] == "RuntimeError"
    assert details["message"] == "boom"
    assert "RuntimeError: boom" in details["traceback"]
