"""URL configuration.

The whole portal *is* the Django admin, so we mount it at the site root: opening
``/`` drops the reviewer straight into the review cockpit. There is no separate
public site to route to.
"""

from __future__ import annotations

from django.contrib import admin
from django.urls import path

urlpatterns = [
    # Admin at the root, not the usual /admin/, because the admin is the app.
    path("", admin.site.urls),
]
