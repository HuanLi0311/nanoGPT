"""Warm Python 3.11 stdlib imports before Ray/vLLM spawn workers start."""

# ponytail: keep the workaround at the interpreter boundary; a vLLM spawn
# child does not run Verl's Ray worker setup hook.  Remove this once the
# Python/Ray/vLLM import race is fixed upstream.
import email.errors  # noqa: F401
import email.feedparser  # noqa: F401
import email.parser  # noqa: F401
import multiprocessing.context  # noqa: F401
import unittest.mock  # noqa: F401
import unittest.result  # noqa: F401
try:
    import zoneinfo._common  # noqa: F401
except ModuleNotFoundError:
    pass
