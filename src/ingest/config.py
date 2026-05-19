from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
NIST_800_53_XLSX_URL = (
    "https://csrc.nist.gov/files/pubs/sp/800/53/r5/upd1/final/docs/"
    "sp800-53r5-control-catalog.xlsx"
)
EPSS_API_URL = "https://api.first.org/data/v1/epss"


@dataclass(frozen=True)
class IngestionConfig:
    raw_data_dir: Path = RAW_DATA_DIR
    external_data_dir: Path = EXTERNAL_DATA_DIR
    cisa_kev_url: str = CISA_KEV_URL
    nist_catalog_url: str = NIST_800_53_XLSX_URL
    epss_api_url: str = EPSS_API_URL
    request_timeout_seconds: int = 30
    cache_ttl_hours: int = 24

    @property
    def cisa_cache_path(self) -> Path:
        return self.external_data_dir / "known_exploited_vulnerabilities.json"

    @property
    def epss_cache_path(self) -> Path:
        return self.external_data_dir / "epss_scores.csv"

    @property
    def nist_cache_path(self) -> Path:
        return self.external_data_dir / "sp800-53r5-control-catalog.xlsx"

