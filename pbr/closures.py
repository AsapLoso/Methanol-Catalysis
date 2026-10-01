"""Closures: the four places where the model can be made more or less detailed.

A packed-bed model is a stack of approximations. Each one is isolated here as
an object with a small interface, so switching it is a change of argument and
not an edit inside the right-hand side. That is the whole design goal: the
right-hand side should be a transcription of the governing equations, with no
branches in it.

The four, and what "more detailed" means for each:

`ExternalTransport`
    Does gas have to cross a film to reach the pellet surface?
    `NoExternalResistance` sets `C_s = C_b` (pseudo-homogeneous).
    `FilmResistance` solves the Robin condition at `r = R_p`.

`RadialTransport`
    Does concentration vary inside a pellet, or is the pellet well-mixed?
    `ExplicitDiffusion` carries `N_r` states per axial cell.
    `Uniform` carries none and sets `c_p = C_s`.

`SurfaceDynamics`
    Does the catalyst remember? `AlgebraicEquilibrium` holds coverages at
    their quasi-steady value; `Relaxation` integrates them toward it with a
    finite time constant, which is the entire reason the model is dynamic.

`AxialDispersion`
    Does the gas back-mix? Off by default: a packed bed at these flow rates is
    plug flow, and with no dispersion the axial problem is hyperbolic.

Reducing every closure to its simplest setting gives a plug-flow, well-mixed
pellet, memoryless-catalyst, purely convective reactor. Turning them all on
gives a dispersed reactor with a resolved pellet and a relaxing surface. Same
right-hand side either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from pbr.species import IADS, Adsorbate

# --- axial ------------------------------------------------------------------


@runtime_checkable
class AxialDispersion(Protocol):
    """Second-order term on the bulk balance, and the boundaries that go with it.

    Dispersion and boundary conditions are one decision, not two. A centred
    second-order stencil without the matching Robin boundaries is unstable, and
    Danckwerts' boundaries exist precisely to supply the inlet and outlet
    fluxes a two-point problem needs. `NoDispersion` therefore does not merely
    set a coefficient to zero -- it also replaces the boundaries with the
    outflow condition a hyperbolic problem actually admits.
    """

    @property
    def coefficient(self) -> float:
        """`D_ax` in m^2/s."""

    def inlet_flux(
        self, c_in: NDArray[np.float64], c_0: NDArray[np.float64], dz: float
    ) -> NDArray[np.float64]:
        """Total molar flow arriving in the first cell, per unit area."""

    def outflow(self, c_n: NDArray[np.float64], dz: float) -> NDArray[np.float64]:
        """Molar flow leaving the last cell, per unit area."""


@dataclass(frozen=True, slots=True)
class NoDispersion:
    """Pure convection. Inlet Dirichlet, zero-gradient outflow.

    The outlet condition is not a free choice. With `D_ax = 0` the axial
    operator is one-sided and its characteristics all point downstream, so the
    only boundary condition that adds no information is the outflow one.
    """

    @property
    def coefficient(self) -> float:
        return 0.0

    def inlet_flux(
        self,
        c_in: NDArray[np.float64],
        c_0: NDArray[np.float64],
        dz: float,
    ) -> NDArray[np.float64]:
        # u_s * C_b(0, t): the Dirichlet value simply enters the balance.
        return c_in

    def outflow(self, c_n: NDArray[np.float64], dz: float) -> NDArray[np.float64]:
        return c_n


@dataclass(frozen=True, slots=True)
class DanckwertsDispersion:
    """Open boundary conditions matched to axial dispersion.

    The inlet supplies `u_s C_in - D_ax dC/dz|_0` and the outlet takes
    `-u_s C_N + D_ax dC/dz|_N`, so the flux entering the domain equals the flux
    leaving it at steady state. Applying a Dirichlet at both ends instead --
    the obvious thing to write -- over-determines the system and pins the exit
    concentration rather than letting it float.
    """

    d_ax: float

    @property
    def coefficient(self) -> float:
        return self.d_ax

    def inlet_flux(
        self,
        c_in: NDArray[np.float64],
        c_0: NDArray[np.float64],
        dz: float,
    ) -> NDArray[np.float64]:
        # Ghost node at -dz with the known inlet value: one-sided derivative.
        derivative = (c_in - c_0) / dz
        return c_in - self.d_ax * derivative

    def outflow(self, c_n: NDArray[np.float64], dz: float) -> NDArray[np.float64]:
        derivative = np.zeros_like(c_n)
        return c_n - self.d_ax * derivative


def dispersion_for(d_ax: float, u_s: float, length: float) -> AxialDispersion:
    """Pick the dispersion closure, refusing an unphysical Peclet.

    Below `Pe = 1` the dispersive term is comparable to or larger than the
    convective one, and the plug-flow picture the rest of the model is built on
    has stopped being true. Rather than quietly returning a wrong answer, this
    refuses -- a value that low means the axial model itself needs revisiting.
    """
    if d_ax <= 0.0:
        return NoDispersion()
    peclet = u_s * length / d_ax
    if peclet < 1.0:
        raise ValueError(
            f"axial dispersion D_ax={d_ax:.3e} m^2/s gives Pe={peclet:.3f} < 1; "
            f"the plug-flow axial model is not valid. Either the geometry or "
            f"the dispersion coefficient is wrong."
        )
    return DanckwertsDispersion(d_ax)


# --- external film ----------------------------------------------------------


@runtime_checkable
class ExternalTransport(Protocol):
    """Boundary condition on the pellet shell, `r = R_p`."""

    def shell_concentration(
        self,
        c_bulk: NDArray[np.float64],
        c_inner: NDArray[np.float64],
        d_eff: float,
        dr: float,
    ) -> NDArray[np.float64]:
        """`c_p` at `r = R_p`, given the bulk gas and the innermost interior node."""


@dataclass(frozen=True, slots=True)
class NoExternalResistance:
    """Pseudo-homogeneous: the gas phase and the pellet surface are one phase.

    This is not an approximation bolted on afterwards -- it is the limit of
    `FilmResistance` as `k_g -> infinity`, and it is correct whenever the
    Carberry number `Ca = r_obs d_p / (k_g C_s)` is small, which for the
    catalyst we use it is by roughly two orders of magnitude.

    Keeping it as its own object rather than as `k_g = inf` means the
    *unlimited* case has a name, which is what the default configuration is
    allowed to say out loud.
    """

    def shell_concentration(
        self,
        c_bulk: NDArray[np.float64],
        c_inner: NDArray[np.float64],
        d_eff: float,
        dr: float,
    ) -> NDArray[np.float64]:
        return c_bulk


@dataclass(frozen=True, slots=True)
class FilmResistance:
    """Finite external film: `D_e dc/dr|_R = k_g (C_b - c_p|_R)`.

    The condition is imposed by eliminating `c_p|_R` algebraically against the
    one-sided gradient, which gives

        c_p|_R = (c_inner + Bi * C_b) / (1 + Bi),    Bi = k_g dr / D_e

    so `Bi -> infinity` recovers the pseudo-homogeneous closure exactly and
    `Bi -> 0` recovers a reflecting shell. No branch, no flag: the two limits
    are the same expression.

    `Bi` here is a *grid* Biot number -- it carries the spacing, not the pellet
    radius. Using `k_g R_p / D_e` would be the mesh-independent one, and both
    appear in the literature; the distinction is easy to lose and changes the
    result, so it is spelled out here.
    """

    k_g: float

    def shell_concentration(
        self,
        c_bulk: NDArray[np.float64],
        c_inner: NDArray[np.float64],
        d_eff: float,
        dr: float,
    ) -> NDArray[np.float64]:
        if self.k_g <= 0.0:
            return c_bulk
        biot = self.k_g * dr / d_eff
        return (c_inner + biot * c_bulk) / (1.0 + biot)


def external_specific_area(particle_radius: float, void_fraction: float) -> float:
    """External surface area per unit bed volume `a_v` [m^-1].

    For spheres, `3 (1 - eps_b) / R_p`: the `(1 - eps_b)` is the solid volume
    fraction, and without it `a_v` is a per-unit-solid quantity being used
    where a per-unit-bed one is required.
    """
    return 3.0 * (1.0 - void_fraction) / particle_radius


# --- radial -----------------------------------------------------------------


@runtime_checkable
class RadialTransport(Protocol):
    """Whether concentration varies inside the pellet, and how finely."""

    @property
    def n_nodes(self) -> int:
        """Radial states carried per axial cell."""

    @property
    def carries_radial_states(self) -> bool:
        """False when the pellet is well mixed and no radial state exists."""

    def radii(self, r_p: float) -> NDArray[np.float64]:
        """Radial node positions [m], from `0` to `r_p` inclusive."""


@dataclass(frozen=True, slots=True)
class ExplicitDiffusion:
    """Pellet resolved in the radial direction.

    State-per-cell is `n_nodes * n_gas`. This is the expensive closure: it is
    the difference between a few dozen unknowns and a few thousand, and it is
    what makes intra-particle diffusion limitation appear as a result rather
    than as an assumption.
    """

    n_radial: int = 24
    geometry: int = 2

    @property
    def n_nodes(self) -> int:
        return self.n_radial

    @property
    def carries_radial_states(self) -> bool:
        return True

    def radii(self, r_p: float) -> NDArray[np.float64]:
        return np.linspace(0.0, r_p, self.n_radial)

    @property
    def dr(self) -> float:  # pragma: no cover - convenience, needs r_p
        raise RuntimeError("dr depends on r_p; call radii() instead")


@dataclass(frozen=True, slots=True)
class Uniform:
    """Well-mixed pellet: `c_p = C_s` everywhere, no radial states at all.

    This is the mathematically correct limit of `ExplicitDiffusion` as
    `D_e -> infinity`, and it is the reason `Uniform` has no tunable
    parameter: there is nothing to tune. Turning it on collapses the state
    vector by a factor of `n_radial * n_species`, which is what makes a
    parameter sweep affordable.
    """

    @property
    def n_nodes(self) -> int:
        return 1

    @property
    def carries_radial_states(self) -> bool:
        return False

    def radii(self, r_p: float) -> NDArray[np.float64]:
        return np.array([r_p])


# --- surface ----------------------------------------------------------------


@runtime_checkable
class SurfaceDynamics(Protocol):
    """Whether the catalyst surface has memory."""

    @property
    def carries_surface_states(self) -> bool:
        """False when the coverages are algebraic and need no state."""

    def derivative(
        self,
        theta: NDArray[np.float64],
        theta_target: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """`dtheta/dt` given actual and quasi-steady coverages."""


@dataclass(frozen=True, slots=True)
class AlgebraicEquilibrium:
    """Coverages sit exactly at their Langmuir value. The pseudo-steady assumption.

    Cheap, always well-posed, and *wrong in a way that matters* for this
    thesis: it cannot represent a surface that is still filling up, so it
    cannot represent the transient selectivity effect the whole project is
    about. It is the default because it is the assumption the steady-state
    literature is calibrated under, and because every conclusion drawn with it
    should be one the calibrated literature already supports.
    """

    @property
    def carries_surface_states(self) -> bool:
        return False

    def derivative(
        self,
        theta: NDArray[np.float64],
        theta_target: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.zeros_like(theta)


@dataclass(frozen=True, slots=True)
class Relaxation:
    """First-order relaxation of the coverages toward their quasi-steady value.

        dtheta/dt = (theta_ss - theta) / tau

    This is a *linearised* dynamic extension of a steady-state rate law, not a
    microkinetic derivation: it says the surface approaches its quasi-steady
    coverage exponentially and names the time constant, without claiming to
    know which elementary step sets it. The honest limitation is that `tau` is
    not read off any source we have, so it is declared as an assumption and
    swept over orders of magnitude before anything is concluded from it.

    The alternative -- integrating real elementary-step coverages -- needs a
    rate constant per step, and the parameter database does not contain them.
    Choosing relaxation over that is a trade of mechanistic detail for a
    parameter we cannot source.
    """

    tau: float

    def __post_init__(self) -> None:
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")

    @property
    def carries_surface_states(self) -> bool:
        return True

    def derivative(
        self,
        theta: NDArray[np.float64],
        theta_target: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return (theta_target - theta) / self.tau


def surface_state_size(dynamics: SurfaceDynamics) -> int:
    """How many scalar coverages a state vector carries, given the closure."""
    if not dynamics.carries_surface_states:
        return 0
    return len(Adsorbate) - 1  # exclude the vacant site, which is 1 - sum


__all__ = [
    "IADS",
    "AlgebraicEquilibrium",
    "AxialDispersion",
    "DanckwertsDispersion",
    "ExplicitDiffusion",
    "ExternalTransport",
    "FilmResistance",
    "NoDispersion",
    "NoExternalResistance",
    "RadialTransport",
    "Relaxation",
    "SurfaceDynamics",
    "Uniform",
    "dispersion_for",
    "external_specific_area",
    "surface_state_size",
]
