"""ASGI entrypoint for the QA portal (for ASGI hosts; runserver uses WSGI)."""

from __future__ import annotations

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "qa_portal.settings")

application = get_asgi_application()
