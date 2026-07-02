"""WSGI entrypoint for the QA portal (used by ``runserver`` and any WSGI host)."""

from __future__ import annotations

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "qa_portal.settings")

application = get_wsgi_application()
