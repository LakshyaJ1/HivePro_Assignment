"""Shared test fixtures and configuration for the TawasolPay risk assistant test suite."""
from __future__ import annotations

import logging

# Suppress noisy third-party loggers during test runs
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
