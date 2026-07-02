"""The review cockpit — a customized Django admin for :class:`Confirmation`.

Two things make the plain admin into a review tool:

1. A changelist tuned for triage: short text preview, whether the row is
   reviewed / corrected / excluded, and filters to slice the queue.
2. A **"Save & next"** button on the change form that stamps the record
   ``reviewed`` and jumps straight to the next unreviewed confirmation, so a
   reviewer can march through the queue without returning to the list.
"""

from __future__ import annotations

from django.contrib import admin
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.utils import timezone

from .models import Confirmation

# POST key set by the custom submit button in change_form.html.
SAVE_AND_NEXT = "_save_and_next"


class CorrectedListFilter(admin.SimpleListFilter):
    """Filter the queue by "does this row have a human correction?".

    ``corrected`` is a JSONField, so there is no ready-made boolean filter; we
    map yes/no to ``corrected IS (NOT) NULL``.
    """

    title = "corrected"
    parameter_name = "corrected"

    def lookups(self, request, model_admin):
        return (("yes", "Yes"), ("no", "No"))

    def queryset(self, request, queryset):
        if self.value() == "yes":
            return queryset.filter(corrected__isnull=False)
        if self.value() == "no":
            return queryset.filter(corrected__isnull=True)
        return queryset


@admin.register(Confirmation)
class ConfirmationAdmin(admin.ModelAdmin):
    """Review cockpit for freight confirmations."""

    # Template adds the "Save & next" button (see handle in response_change).
    change_form_template = "admin/review/confirmation/change_form.html"

    list_display = ("id", "short_text", "reviewed", "has_correction", "exclude")
    list_filter = ("reviewed", "exclude", CorrectedListFilter)
    search_fields = ("text",)
    readonly_fields = ("created",)
    list_per_page = 50

    @admin.display(description="text")
    def short_text(self, obj: Confirmation) -> str:
        """One-line preview of the raw confirmation for the changelist."""
        flat = obj.text.replace("\n", " ")
        return (flat[:70] + "…") if len(flat) > 70 else flat

    @admin.display(boolean=True, description="corrected")
    def has_correction(self, obj: Confirmation) -> bool:
        """Green check in the changelist when a reviewer has fixed the parse."""
        return obj.corrected is not None

    # -- the "mark reviewed and next" flow ---------------------------------

    def response_change(self, request, obj):
        """Handle the "Save & next" submit button.

        The base admin has already validated the form and saved ``obj`` (via
        ``save_model``) by the time this runs, so any correction the reviewer
        typed is persisted. We then stamp the record ``reviewed`` and redirect
        to the change page of the next unreviewed confirmation — the classic
        review-queue flow. Any other submit button falls through to Django's
        default behavior.
        """
        if SAVE_AND_NEXT in request.POST:
            obj.reviewed = timezone.now()
            obj.save(update_fields=["reviewed"])

            nxt = self._next_unreviewed(after_pk=obj.pk)
            if nxt is not None:
                self.message_user(
                    request,
                    f"Marked #{obj.pk} reviewed. Now reviewing #{nxt.pk}.",
                )
                url = reverse("admin:review_confirmation_change", args=[nxt.pk])
                return HttpResponseRedirect(url)

            # Nothing left in the queue — back to the (now empty) changelist.
            self.message_user(
                request,
                f"Marked #{obj.pk} reviewed. No more unreviewed confirmations.",
            )
            url = reverse("admin:review_confirmation_changelist")
            return HttpResponseRedirect(url)

        return super().response_change(request, obj)

    @staticmethod
    def _next_unreviewed(after_pk: int) -> Confirmation | None:
        """The next confirmation still awaiting review.

        Prefer the next one *after* the current pk (natural forward march); if
        the reviewer just cleared the tail of the queue, wrap to the first
        unreviewed row anywhere. Returns ``None`` when the queue is empty.
        """
        forward = (
            Confirmation.objects.filter(reviewed__isnull=True, pk__gt=after_pk)
            .order_by("pk")
            .first()
        )
        if forward is not None:
            return forward
        return (
            Confirmation.objects.filter(reviewed__isnull=True)
            .order_by("pk")
            .first()
        )
