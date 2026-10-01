"""Provenance for every number the model is allowed to use.

The workspace rules forbid inventing a constant (Rule 16). Enforcement is
stronger here than a comment: each parameter carries a `Provenance` record,
and a registry lookup for an unregistered name raises rather than returning
a default. A number that nobody sourced cannot enter the model.

Three kinds, in decreasing order of trust:

`Literature`
    Read off a primary source. Cite the primary source -- never an internal
    extraction document, which is a working note and not citable.
`Derived`
    Computed from other parameters by a formula that is written down here
    (an ideal-gas conversion, a quadrature weight, a derived Thiele modulus).
`Assumed`
    Chosen by us because nothing in the literature fixes it. These are the
    dangerous ones, so they must be swept before any conclusion is drawn that
    depends on them. `Assumed` parameters are listed by `assumed_parameters`
    so a sensitivity study cannot silently skip one.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class Source(StrEnum):
    """How much we trust a number."""

    LITERATURE = "literature"
    """Read directly off a primary source."""

    DERIVED = "derived"
    """Computed from other parameters by a formula recorded in the docstring."""

    ASSUMED = "assumed"
    """Chosen by us. Requires a sensitivity study before it is relied upon."""


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where one number came from.

    `reference` is a citation key into `references.bib`. `locator` is the
    precise spot inside that source -- a table, an equation, a page -- because
    "see the paper" is not traceable and traceability is the entire point.
    """

    reference: str
    locator: str
    source: Source = Source.LITERATURE
    note: str = ""

    def __str__(self) -> str:
        tail = f" -- {self.note}" if self.note else ""
        return f"{self.reference} ({self.locator}){tail}"


class UnknownParameter(KeyError):
    """Raised when the model asks for a parameter nobody sourced."""


@dataclass(frozen=True, slots=True)
class Registry(Mapping[str, Provenance]):
    """Name -> provenance, with a hard failure on an unregistered name.

    Subclasses `Mapping` so `dict(registry)` and `in` work, but `__getitem__`
    raises `UnknownParameter` instead of `KeyError`, because a missing
    parameter is a modelling error rather than a lookup miss.
    """

    _entries: Mapping[str, Provenance] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Provenance:
        try:
            return self._entries[name]
        except KeyError:
            raise UnknownParameter(
                f"{name!r} has no provenance record, so it is not in the model. "
                f"Add it to the registry with a citation, or derive it from "
                f"parameters that are registered."
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def assumed_parameters(self) -> list[str]:
        """Names whose value we chose rather than read. Must be swept."""
        return sorted(
            name
            for name, prov in self._entries.items()
            if prov.source is Source.ASSUMED
        )

    def uncited(self) -> list[str]:
        """Names with a `reference` but no `locator`. Usually a mistake."""
        return sorted(
            name for name, prov in self._entries.items() if not prov.locator.strip()
        )
