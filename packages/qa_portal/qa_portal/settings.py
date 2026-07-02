"""Django settings for the freight-parser QA portal.

This is a deliberately *minimal* Django project. Its only job is to give a human
reviewer a cockpit (the built-in admin) for correcting the model's auto-parses
before they become training data. Everything is laptop-local: SQLite on disk, no
external services, no auth beyond the demo superuser you create with
``createsuperuser``.
"""

from __future__ import annotations

from pathlib import Path

# packages/qa_portal/qa_portal/settings.py -> parents: .parent == qa_portal/,
# .parent.parent == packages/qa_portal/. The SQLite file and manage.py live here.
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# DEMO ONLY. This SECRET_KEY is hard-coded and public on purpose: the portal is
# a teaching tool that runs on your laptop, never on the open internet. NEVER
# reuse this key (or this settings module) for a real deployment.
# ---------------------------------------------------------------------------
SECRET_KEY = "django-insecure-pyohio-freight-demo-key-not-for-production-use"

DEBUG = True
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # The review cockpit — our one app.
    "qa_portal.review",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "qa_portal.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        # APP_DIRS lets Django find our admin change_form.html override under
        # qa_portal/review/templates/admin/review/confirmation/.
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "qa_portal.wsgi.application"
ASGI_APPLICATION = "qa_portal.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
        # Tests run against an in-memory database so they never touch the real
        # review DB and leave no artifacts on disk.
        "TEST": {"NAME": ":memory:"},
    }
}

# Demo-only: no password validators so a reviewer can create a trivial local
# superuser without ceremony. Do not copy for anything real.
AUTH_PASSWORD_VALIDATORS: list[dict] = []

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
