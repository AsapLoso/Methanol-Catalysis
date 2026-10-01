"""Sweeping the parameters the model had to choose for itself.

Six numbers in this model are not read off a paper: the Larsen rate cap, three
pellet properties, the bed void fraction, and the surface relaxation time.
Each is a legitimate modelling decision, and each is a place where a
conclusion could be an artefact of the choice rather than a result.

This module makes that checkable instead of a promise. `assumed_sweep` returns
the grid of alternatives for every `ASSUMED` provenance entry,
`sweep_variation` maps a provenance key to the object it lives in so the grid
can actually be applied, and `audit_assumptions` reports which of the six are
reachable from a given `Inputs` -- so a run that never touches the Larsen cap
is not silently credited with having swept it.

The test `tests/test_sweep.py::test_every_assumed_parameter_is_sweepable` fails
if anyone adds an `ASSUMED` provenance entry without a corresponding sweep and
an application path. That is the whole mechanism: an unsourced number cannot
enter the model without also entering a list of numbers it has to be checked
against.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

from pbr import parameter_sets as ps
from pbr.closures import Relaxation
from pbr.model import Inputs

#: provenance key -> the candidate values to run.
#: Mirrors `parameter_sets.ASSUMED_SWEEP`; the duplicate is deliberate because
#: one is the record of what was assumed and the other is the record of what
#: was checked, and they are allowed to disagree only if a sweep has been
#: dropped on purpose.
SWEEP_GRID: dict[str, list[float]] = dict(ps.ASSUMED_SWEEP)


@dataclass(frozen=True, slots=True)
class Variation:
    """One provenance key and the value it takes in a given `Inputs`."""

    key: str
    value: float

    def apply(self, inputs: Inputs) -> Inputs:
        """Return a copy of `inputs` with this value substituted.

        Raises rather than silently skipping when the key does not reach this
        configuration. A sweep that quietly ignores the parameter it is
        supposed to be varying is worse than no sweep: it reports robustness
        that was never tested.
        """
        if self.key == "surface.tau":
            return replace(
                inputs,
                surface=Relaxation(tau=self.value),
            )
        if self.key.startswith("catalyst."):
            field = self.key.split(".", 1)[1]
            if not hasattr(inputs.catalyst, field):
                raise KeyError(f"{self.key} does not exist on Catalyst")
            return replace(
                inputs,
                catalyst=replace(inputs.catalyst, **{field: self.value}),
            )
        if self.key.startswith("geometry."):
            field = self.key.split(".", 1)[1]
            if not hasattr(inputs.geometry, field):
                raise KeyError(f"{self.key} does not exist on Geometry")
            return replace(
                inputs,
                geometry=replace(inputs.geometry, **{field: self.value}),
            )
        if self.key.startswith("larsen."):
            field = self.key.split(".", 1)[1]
            kinetics = replace(inputs.kinetics, **{field: self.value})
            return replace(inputs, kinetics=kinetics)
        raise KeyError(
            f"no substitution path for {self.key!r}; add one so every assumed "
            f"parameter is actually sweepable"
        )


def assumed_sweep(inputs: Inputs) -> dict[str, list[float]]:
    """The sweep grid restricted to parameters this configuration can reach.

    The Larsen cap only exists in the Larsen set and the relaxation time only
    matters when the surface closure carries states, so a Urnes run with an
    algebraic surface genuinely has nothing to sweep from those two.
    """
    reachable: dict[str, list[float]] = {}
    for key, values in SWEEP_GRID.items():
        if key.startswith("larsen.") and not hasattr(inputs.kinetics, "r_1max"):
            continue
        if key == "surface.tau" and not inputs.surface.carries_surface_states:
            continue
        reachable[key] = list(values)
    return reachable


def iter_variations(inputs: Inputs) -> Iterator[tuple[str, float, Inputs]]:
    """Yield `(key, value, varied_inputs)` for every reachable combination."""
    for key, values in assumed_sweep(inputs).items():
        for value in values:
            yield key, value, Variation(key, value).apply(inputs)


def audit_assumptions(inputs: Inputs) -> dict[str, Any]:
    """Reachable sweep grid, and the assumed entries it does not reach."""
    reachable = assumed_sweep(inputs)
    all_assumed = set(ps.PROVENANCE.assumed_parameters())
    return {
        "reachable": reachable,
        "unreachable": sorted(all_assumed - set(reachable)),
        "n_runs": sum(len(v) for v in reachable.values()),
    }


def run_sweep(inputs: Inputs, metric) -> dict[str, Any]:
    """Apply the sweep and collect `metric(inputs) -> float` for each run.

    `metric` is a callable taking an `Inputs` and returning a number. Keeping
    the metric outside means the sweep is reusable: conversion, selectivity,
    residence-time spread and pressure drop all want the same grid and none of
    them should re-implement it.
    """
    rows: list[dict[str, Any]] = []
    for key, value, varied in iter_variations(inputs):
        rows.append(
            {
                "parameter": key,
                "value": value,
                "metric": metric(varied),
            }
        )
    values = [row["metric"] for row in rows]
    return {
        "rows": rows,
        "n_runs": len(rows),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "spread": (max(values) - min(values)) if values else None,
    }


__all__ = [
    "SWEEP_GRID",
    "Variation",
    "assumed_sweep",
    "audit_assumptions",
    "iter_variations",
    "run_sweep",
]
