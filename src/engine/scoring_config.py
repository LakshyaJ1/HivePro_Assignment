from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScoreWeights:
    """Composite Risk Score weights.

    The default weights intentionally keep CVSS at 20% so high base severity
    does not dominate business exposure, ransomware, and active exploitation.
    """

    cvss: float = 0.20
    active_exploitation: float = 0.15
    ransomware: float = 0.20
    epss: float = 0.15
    internet_exposed: float = 0.10
    business_impact: float = 0.10
    threat_intel_match: float = 0.05
    days_open: float = 0.03
    missing_edr: float = 0.02

    def validate(self) -> None:
        total = sum(self.as_dict().values())
        if abs(total - 1.0) > 0.000001:
            raise ValueError(f"Score weights must sum to 1.0, got {total:.6f}")
        for name, value in self.as_dict().items():
            if value < 0:
                raise ValueError(f"Score weight {name} cannot be negative")

    def as_dict(self) -> dict[str, float]:
        return {
            "cvss": self.cvss,
            "active_exploitation": self.active_exploitation,
            "ransomware": self.ransomware,
            "epss": self.epss,
            "internet_exposed": self.internet_exposed,
            "business_impact": self.business_impact,
            "threat_intel_match": self.threat_intel_match,
            "days_open": self.days_open,
            "missing_edr": self.missing_edr,
        }


@dataclass(frozen=True)
class SeverityBand:
    label: str
    minimum_score: float


@dataclass(frozen=True)
class ScoringConfig:
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    top_n: int = 5
    days_open_full_penalty: int = 90
    stale_asset_days: int = 30
    severity_bands: tuple[SeverityBand, ...] = (
        SeverityBand("Critical", 0.75),
        SeverityBand("High", 0.60),
        SeverityBand("Medium", 0.40),
        SeverityBand("Low", 0.0),
    )

    def __post_init__(self) -> None:
        self.weights.validate()
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")
        if self.days_open_full_penalty < 1:
            raise ValueError("days_open_full_penalty must be >= 1")

