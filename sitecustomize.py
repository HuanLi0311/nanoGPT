"""Warm Python 3.11 stdlib imports before Ray/vLLM spawn workers start."""

# ponytail: keep the workaround at the interpreter boundary; a vLLM spawn
# child does not run Verl's Ray worker setup hook.  Remove this once the
# Python/Ray/vLLM import race is fixed upstream.
import importlib


def _warm(name: str) -> None:
    try:
        importlib.import_module(name)
    except (ImportError, AttributeError):
        # Some supported Python versions do not have every private module;
        # another pass can finish a package whose parent was half-imported.
        pass


for _ in range(2):
    for _module in (
        "urllib.parse",
        "json",
        "json.decoder",
        "email.errors",
        "email.feedparser",
        "email.parser",
        "multiprocessing.context",
        "unittest.mock",
        "unittest.result",
        "zoneinfo._common",
    ):
        _warm(_module)
