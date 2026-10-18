---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Parameters and provenance

Every quantity the model consumes is declared once, as a typed schema, before anything uses it.
Nothing downstream is allowed to carry a bare number that did not come out of one of those
declarations. That is what makes the model auditable from the outside, and it is what lets the
test suite walk the registry and fail on a unit nobody checked.

## Why units are handled where they are

The sources disagree. One prints kPa, another bar, a third Pa. Rather than requiring every call
site to remember which is which, each kinetic parameter set converts **on the way in**, and the
conversion factor is part of that set's own declaration. A rate constant therefore cannot be
used with the wrong pressure scale, because the conversion is not a call-site decision.

The model works in SI throughout: metres, seconds, kilograms, moles, kelvin, pascals, with
partial pressures in pascals.

## Why parameter sets are types

Each kinetic literature source is its own schema. Switching between them is a **type change**,
not a set of magic numbers scattered through the code. That means:

- A parameter that exists in one source and not another cannot be silently borrowed.
- A missing value is a construction error, not a `None` that fails later inside an integrator.
- The units of a set travel with the set.

## Provenance

Each parameter set records where it came from and what was converted. Where a source reports a
composite rate constant — a product of elementary constants reported as one number — that is
what is stored, and the model does not attempt to decompose it. The consequence is stated in
the Theory chapter: the model cannot reconstruct elementary surface kinetics from these numbers,
and it does not pretend to.

Where a source reports adsorption terms individually, coverages *can* be written down, and that
set is the one that supports genuine transient surface dynamics.

## What is deliberately not parameterised

Some quantities that appear in the equations are not fitted parameters at all: the gas constant,
the molar mass of the active metal, and the Faraday constant are fixed physics, held separately
from the modelling parameters so that a change in a physical constant cannot be mistaken for a
change in a fitted value.
