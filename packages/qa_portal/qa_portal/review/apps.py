"""App configuration for the review cockpit."""

from __future__ import annotations

from django.apps import AppConfig


class ReviewConfig(AppConfig):
    """The one app in the QA portal.

    ``name`` is the dotted import path (``qa_portal.review``); Django derives the
    short app *label* ``review`` from the last component, which is what the
    admin URL names (``admin:review_confirmation_change``) and
    ``makemigrations review`` use.
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "qa_portal.review"
    label = "review"
    verbose_name = "Confirmation review"
