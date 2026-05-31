"""Per-tag suggestion toggle dialog.

Lets the user pick which mechanic categories should produce a "press this"
recommendation vs which should fall through to the spell's default phrase.
Example: keep snare/bleed recommendations on (so Freedom and BoP get
called), turn big_damage_party off (you just want to hear the spell name,
not be told to press Aura Mastery yet again).

The checkbox list is **derived from the loaded TagRules.precedence** —
adding a tag to `config/tag_rules.yaml` automatically surfaces it here on
the next launch. No second source of truth to maintain.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from wow_alert.tag_rules import TagRules


# Short description per tag so users know what triggers each. Keys must
# match the canonical lowercase tag tokens used in tag_rules.yaml. Missing
# entries fall back to a "(no description)" placeholder.
_TAG_DESCRIPTIONS: dict[str, str] = {
    "interrupt": "Interruptible cast → recommend a kick / stun.",
    "stop": "Non-interruptible channel → recommend a knockback or stun.",
    "cc": "An add the party should crowd-control → stun / disorient.",
    "big_damage_single": "Hard hit on one player → external defensive.",
    "big_damage_party": "Hard hit on the whole party → party-wide wall.",
    "bleed": "Bleed effect → BoP / aggro-dropping external.",
    "dispel_magic": "Magic debuff → cleanse.",
    "dispel_curse": "Curse → cleanse-style dispel.",
    "dispel_poison": "Poison → cleanse-style dispel.",
    "dispel_disease": "Disease → cleanse-style dispel.",
    "snare": "Player rooted/snared → Freedom.",
    "dodge": "Avoidable swirly → just dodge it; no callout action.",
}


class TagSuggestionsDialog(QDialog):
    """Pick which tag categories get press-this-button callouts."""

    def __init__(
        self,
        tag_rules: TagRules,
        enabled_tags: set[str] | None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Suggestion categories")
        self.resize(440, 460)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        intro = QLabel(
            "Pick which mechanic categories should produce a "
            "'press this button' callout. Unchecked categories play just "
            "the spell's plain alert phrase instead. Per-spell rules are "
            "unaffected — they're authored deliberately and always fire."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        # Build checkboxes from the loaded tag_rules.precedence list — the
        # canonical tag set. No second source of truth.
        tags = list(tag_rules.precedence)
        # Default: everything enabled.
        effective_enabled = (
            set(tags) if enabled_tags is None else set(enabled_tags)
        )
        self._checks: dict[str, QCheckBox] = {}
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        for tag in tags:
            row_widget = QWidget()
            row_layout = QVBoxLayout(row_widget)
            row_layout.setContentsMargins(2, 2, 2, 2)
            row_layout.setSpacing(0)
            checkbox = QCheckBox(tag)
            checkbox.setChecked(tag in effective_enabled)
            desc = QLabel(_TAG_DESCRIPTIONS.get(tag, "(no description)"))
            desc.setStyleSheet("color: gray; margin-left: 20px;")
            desc.setWordWrap(True)
            row_layout.addWidget(checkbox)
            row_layout.addWidget(desc)
            col.addWidget(row_widget)
            self._checks[tag] = checkbox
        col.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        root.addWidget(scroll, stretch=1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def result_enabled_tags(self) -> set[str]:
        """The set of tags the user left checked.

        Returned as a set rather than the sentinel "all enabled" None,
        because once the user has opened this dialog at least once their
        intent is explicit. The caller persists it to QSettings and pushes
        it to the engine via `set_enabled_tags`.
        """
        return {tag for tag, checkbox in self._checks.items() if checkbox.isChecked()}
