"""pytest bootstrap for the QA portal without pytest-django.

We configure Django by hand: point ``DJANGO_SETTINGS_MODULE`` at the portal's
settings and call ``django.setup()`` at import time so plain-function tests can
import models. A session-scoped autouse fixture then builds an **in-memory**
SQLite test database (settings pin ``TEST NAME = ':memory:'``) by running the
migrations, so DB-backed tests — including Django ``TestCase`` subclasses, which
wrap each test in a rolled-back transaction — have a schema to talk to and leave
nothing on disk.
"""

from __future__ import annotations

import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "qa_portal.settings")
django.setup()

import pytest  # noqa: E402  (must follow django.setup())
from django.db import connection  # noqa: E402
from django.test.utils import setup_test_environment, teardown_test_environment  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _test_database():
    """Create/destroy the in-memory test DB once for the whole session."""
    setup_test_environment()
    old_config = connection.creation.create_test_db(verbosity=0, autoclobber=True)
    try:
        yield
    finally:
        connection.creation.destroy_test_db(old_config, verbosity=0)
        teardown_test_environment()
