"""The one table in the QA portal: a :class:`Confirmation` awaiting review.

Each row pairs a raw freight confirmation with the model's *auto-parse*
(``parsed``) and, once a human has looked at it, an optional *correction*
(``corrected``). The label that eventually feeds training is whichever of those
two the reviewer trusts — see :pyattr:`Confirmation.training_label`.

The point of the portal (and of this model's :meth:`clean`) is that a human
never hand-edits free JSON into an invalid shape: every ``parsed``/``corrected``
value is validated against the shared :class:`freight_schema.ParsedConfirmation`
before it can be saved through a form.
"""

from __future__ import annotations

from django.core.exceptions import ValidationError
from django.db import models

from freight_schema import ParsedConfirmation


class Confirmation(models.Model):
    """A single freight confirmation flowing through review.

    Fields:
        text: the raw confirmation message, exactly as received.
        parsed: the model's auto-parse — a JSON list of ShipmentLine dicts.
        corrected: the reviewer's fix, if any (same shape); null until edited.
        reviewed: timestamp a reviewer signed off; null means "still in queue".
        exclude: reviewer's "drop this from training" flag (e.g. junk input).
        created: when the row was loaded/created.
    """

    text = models.TextField(help_text="Raw shipment-confirmation text, as received.")

    # The auto-parse. A JSON list of ShipmentLine dicts (see freight_schema).
    parsed = models.JSONField(help_text="Model auto-parse: a list of ShipmentLine dicts.")

    # The human correction. Null/blank until a reviewer edits the parse.
    corrected = models.JSONField(
        null=True,
        blank=True,
        help_text="Reviewer's corrected parse (list of ShipmentLine dicts); blank if the auto-parse was fine.",
    )

    # Set when a reviewer signs off. Null == still unreviewed (in the queue).
    reviewed = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When a reviewer signed off. Blank means still in the review queue.",
    )

    # Reviewer's "this example is bad, keep it out of training" switch.
    exclude = models.BooleanField(default=False, help_text="drop from training")

    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]

    def __str__(self) -> str:  # pragma: no cover - trivial display helper
        preview = self.text[:60].replace("\n", " ")
        return f"Confirmation #{self.pk}: {preview}"

    # -- validation ---------------------------------------------------------

    def clean(self) -> None:
        """Reject any ``parsed``/``corrected`` value that is not a valid parse.

        Django calls this from ``full_clean()`` (which the admin runs on save),
        so a reviewer can never persist JSON that would break downstream
        training. We validate against the *shared* schema, so "valid here" means
        exactly "valid everywhere else in the pipeline".
        """
        super().clean()
        self._validate_schema("parsed", self.parsed)
        # corrected is optional; only validate it when the reviewer supplied one.
        if self.corrected is not None:
            self._validate_schema("corrected", self.corrected)

    @staticmethod
    def _validate_schema(field: str, value: object) -> None:
        """Validate one JSON value as a list of ShipmentLine records.

        Raises a field-scoped :class:`~django.core.exceptions.ValidationError`
        (so the admin highlights the offending field) if pydantic rejects it.
        """
        try:
            # ParsedConfirmation is a RootModel over list[ShipmentLine]; passing
            # the raw list validates every line and every field type at once.
            ParsedConfirmation.model_validate(value)
        except Exception as exc:  # pydantic ValidationError (or a type error)
            raise ValidationError(
                {field: f"Not a valid list of ShipmentLine records: {exc}"}
            ) from exc

    # -- derived label ------------------------------------------------------

    @property
    def training_label(self) -> object:
        """The parse we trust: the reviewer's ``corrected`` if present, else ``parsed``.

        This is the single place that encodes "a correction wins over the
        auto-parse", so the admin display and the training export agree.
        """
        return self.corrected if self.corrected is not None else self.parsed
