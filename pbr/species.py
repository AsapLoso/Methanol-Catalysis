"""The chemical species of the semi-hydrogenation network, and their algebra.

Four gas species and three adsorbed ones. The choice is not cosmetic: it is
what makes an element balance check possible, and an element balance is the
cheapest test that a stoichiometry table has been transcribed correctly.

Ethane is deliberately absent from the adsorbed set. It leaves the surface
immediately -- there is no adsorbed ethane intermediate in the literature we
parameterise from, and inventing one would be an unsourced mechanistic claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class Gas(StrEnum):
    """Gas-phase species, in the order used by every state array."""

    H2 = "H2"
    C2H2 = "C2H2"
    C2H4 = "C2H4"
    C2H6 = "C2H6"


class Adsorbate(StrEnum):
    """Adsorbed species occupying the surface.

    `FREE` is the vacant site. It is stored so that the site balance is an
    identity rather than an assumption: `sum(theta) == 1` is then checkable.
    """

    FREE = "*"
    H = "H*"
    C2H2 = "C2H2*"
    C2H4 = "C2H4*"


@dataclass(frozen=True, slots=True)
class Formula:
    """Element counts. Used for the atom-balance test, nothing else."""

    counts: dict[str, int]

    def __add__(self, other: Formula) -> Formula:
        merged = dict(self.counts)
        for element, n in other.counts.items():
            merged[element] = merged.get(element, 0) + n
        return Formula(merged)

    def __sub__(self, other: Formula) -> Formula:
        merged = dict(self.counts)
        for element, n in other.counts.items():
            merged[element] = merged.get(element, 0) - n
        return Formula(merged)

    def __mul__(self, factor: float) -> Formula:
        return Formula({element: factor * n for element, n in self.counts.items()})

    def __rmul__(self, factor: float) -> Formula:
        return self * factor

    def is_zero(self, tol: float = 1e-12) -> bool:
        return all(abs(n) <= tol for n in self.counts.values())


FORMULAS: dict[str, Formula] = {
    Gas.H2: Formula({"C": 0, "H": 2}),
    Gas.C2H2: Formula({"C": 2, "H": 2}),
    Gas.C2H4: Formula({"C": 2, "H": 4}),
    Gas.C2H6: Formula({"C": 2, "H": 6}),
    Adsorbate.H: Formula({"H": 1}),
    Adsorbate.C2H2: Formula({"C": 2, "H": 2}),
    Adsorbate.C2H4: Formula({"C": 2, "H": 4}),
}

#: Molar mass in kg/mol, and the number of atoms each molecule carries per mole.
MOLAR_MASS: dict[str, float] = {
    Gas.H2: 2.016e-3,
    Gas.C2H2: 26.038e-3,
    Gas.C2H4: 28.054e-3,
    Gas.C2H6: 30.070e-3,
}

GASES: tuple[str, ...] = tuple(Gas)
ADSORBATES: tuple[str, ...] = tuple(Adsorbate)

#: Column indices, so callers never hand-code `3` for ethane.
IGAS: dict[str, int] = {name: i for i, name in enumerate(GASES)}
IADS: dict[str, int] = {name: i for i, name in enumerate(ADSORBATES)}

#: Element totals of the whole gas mixture, for the atom-balance diagnostic.
#: Both hydrogen and carbon must be conserved along a trajectory.
ELEMENTS: tuple[str, ...] = ("C", "H")


def atom_counts(
    n_gas: Mapping[str, float], n_ads: Mapping[str, float]
) -> tuple[float, ...]:
    """Atoms of each element in the gas and the adsorbed phase.

    `n_gas` maps a gas species name to a mole count; `n_ads` does the same for
    an adsorbate. Adsorbed carbon and hydrogen are counted at face value: an
    adsorbed intermediate carries real atoms that came out of the gas phase and
    will go back into it, so excluding them would make the balance look broken
    for exactly the transient where it matters.
    """
    total = {element: 0.0 for element in ELEMENTS}
    for name, moles in n_gas.items():
        for element, count in FORMULAS[name].counts.items():
            total[element] += moles * count
    for name, moles in n_ads.items():
        for element, count in FORMULAS[name].counts.items():
            total[element] += moles * count
    return tuple(total[element] for element in ELEMENTS)
