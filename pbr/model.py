"""Assembling a runnable model from parameters and closures.

Construction is separated from use, deliberately. `build_model` reads
parameters and returns a frozen `Model`; nothing downstream constructs
anything, so a test can build the model once and hand the same object to
several different integrators, and a reader can see the complete list of what
went into a run in one place.

This module is also where the defaults live, and the defaults are the model's
actual scientific position:

* **No external film resistance.** The measured Carberry number at this
  catalyst is ~3e-4 against a 0.05 threshold, so the film is inactive by a
  factor of 150. Turning it on is available and tested; it is simply not what
  the data supports.
* **No axial dispersion.** Plug flow is the intended regime, and the axial
  problem stays hyperbolic, which is why the outflow boundary condition is
  zero-gradient rather than Danckwerts.
* **Intra-particle diffusion resolved.** This one is *on* by default, unlike
  the two above, because it is not an approximation being made for
  convenience -- it is the reason the pellet is resolved at all, and switching
  it off is the well-mixed-pellet limit rather than the reverse.
* **Memoryless surface.** The quasi-steady-state assumption is the default
  because it is what the steady-state literature is calibrated under. It is
  also the assumption the transient study is testing, so `Relaxation` is
  one argument away.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from pbr.chemistry import rate_law
from pbr.closures import (
    AlgebraicEquilibrium,
    ExplicitDiffusion,
    NoExternalResistance,
    RadialTransport,
    SurfaceDynamics,
    dispersion_for,
)
from pbr.discretisation import FiniteVolumeRadialGrid
from pbr.parameters import (
    DIFFUSION_VOLUMES,
    MASSES,
    Catalyst,
    Geometry,
    KineticSet,
    OperatingPoint,
    SolverSettings,
    fuller_diffusivity,
)
from pbr.solver import Layout, Model, derive
from pbr.species import GASES


def effective_diffusivities(
    catalyst: Catalyst, operating: OperatingPoint
) -> dict[str, float]:
    """`D_e = eps_p D_m / tau` for each species, via a gas-phase correlation.

    Porosity-inclusive, matching the convention declared in
    `TransportProperties`. H2 diffuses an order of magnitude faster than the
    hydrocarbons, and the ratio matters: the hydrogenation steps are
    hydrogen-limited, so a model that used one diffusivity for everything would
    over-predict the rate at the pellet surface and under-predict the internal
    limitation at the same time.
    """
    carrier = operating.inert or "C2H4"
    out: dict[str, float] = {}
    for species in GASES:
        d_molecular = fuller_diffusivity(
            temperature=operating.temperature,
            pressure=operating.pressure,
            mass_a=MASSES[species],
            mass_b=MASSES[carrier],
            vol_a=DIFFUSION_VOLUMES[species],
            vol_b=DIFFUSION_VOLUMES[carrier],
        )
        out[species] = catalyst.particle_porosity * d_molecular / catalyst.tortuosity
    return out


@dataclass(frozen=True, slots=True)
class Inputs:
    """The complete parameter set of a run.

    `solver.n_radial` is the single source of truth for the radial resolution.
    The radial *closure* is asked only whether it carries radial states at all,
    never how many: two places holding that number is how a run ends up with a
    24-node grid described as a 20-node one, which is invisible until a state
    count looks wrong.
    """

    geometry: Geometry
    catalyst: Catalyst
    operating: OperatingPoint
    kinetics: KineticSet
    solver: SolverSettings
    radial: RadialTransport
    external: object
    surface: SurfaceDynamics
    axial_dispersion: float = 0.0

    @staticmethod
    def catalogued(name: str = "urmes2019") -> Inputs:
        """A complete, sourced parameter set by name."""
        from pbr import parameter_sets as ps

        if name == "urmes2019":
            return Inputs(
                geometry=ps.urmes_geometry(),
                catalyst=ps.urmes_catalyst(),
                operating=ps.urmes_operating(),
                kinetics=ps.urmes_2019(),
                solver=SolverSettings(),
                radial=ExplicitDiffusion(geometry=2),
                external=NoExternalResistance(),
                surface=AlgebraicEquilibrium(),
            )
        if name == "borodzinski2000":
            return Inputs(
                geometry=ps.urmes_geometry(),
                catalyst=ps.urmes_catalyst(),
                operating=ps.borodzinski_operating(),
                kinetics=ps.borodzinski_2000(),
                solver=SolverSettings(),
                radial=ExplicitDiffusion(geometry=2),
                external=NoExternalResistance(),
                surface=AlgebraicEquilibrium(),
            )
        if name == "larsen2024":
            return Inputs(
                geometry=ps.urmes_geometry(),
                catalyst=ps.larsen_catalyst(),
                operating=ps.larsen_operating(),
                kinetics=ps.larsen_2024(),
                solver=SolverSettings(),
                radial=ExplicitDiffusion(geometry=2),
                external=NoExternalResistance(),
                surface=AlgebraicEquilibrium(),
            )
        raise KeyError(f"no catalogued parameter set named {name!r}")


def build_model(
    inputs: Inputs,
    inlet,
    *,
    diffusivities: dict[str, float] | None = None,
) -> Model:
    """Assemble the `Model`. Creation only; it integrates nothing."""
    geometry = inputs.geometry
    catalyst = inputs.catalyst
    operating = inputs.operating
    solver = inputs.solver

    derived = derive(
        n_z=solver.n_axial,
        tube_inner_diameter=geometry.tube_inner_diameter,
        catalyst_mass=geometry.catalyst_mass,
        void_fraction=geometry.bed_void_fraction,
        particle_density=catalyst.particle_density,
        particle_porosity=catalyst.particle_porosity,
        volumetric_flow=operating.volumetric_flow,
    )

    d_eff = diffusivities or effective_diffusivities(catalyst, operating)

    n_radial = solver.n_radial
    carries_radial = inputs.radial.carries_radial_states
    carries_surface = inputs.surface.carries_surface_states

    layout = Layout(
        n_z=solver.n_axial,
        n_gas=len(GASES),
        n_radial=n_radial,
        carries_radial=carries_radial,
        carries_surface=carries_surface,
    )

    grid = FiniteVolumeRadialGrid(
        r_p=catalyst.particle_radius,
        n_nodes=n_radial if carries_radial else 1,
        geometry=solver.radial_geometry,
    )

    law = rate_law(
        inputs.kinetics,
        site_basis=catalyst.metal_site_density or 1.0,
        temperature=operating.temperature,
    )

    return Model(
        layout=layout,
        radial=inputs.radial,
        external=inputs.external,
        surface=inputs.surface,
        dispersion=dispersion_for(
            inputs.axial_dispersion,
            derived.u_superficial,
            derived.bed_length,
        ),
        rate=law,
        grid=grid,
        derived=derived,
        temperature=operating.temperature,
        diffusivities=d_eff,
        advection_scheme=str(solver.advection_scheme),
        inlet=inlet,
    )


# --- inlets ----------------------------------------------------------------


def constant_inlet(operating: OperatingPoint) -> NDArray[np.float64]:
    """Steady feed as a `(n_gas,)` concentration vector."""
    concentrations = operating.concentrations()
    return np.array([concentrations.get(name, 0.0) for name in GASES], dtype=float)


def make_inlet(
    operating: OperatingPoint,
    forcing: Sequence[tuple[float, float]] | None = None,
) -> callable:
    """Inlet concentration as a function of time.

    `forcing` is a list of `(frequency_hz, relative_amplitude)` sine terms
    superposed on the steady feed, summed in phase at `t = 0`. Multisine
    excitation is the measurement this thesis is built around, so it is a
    first-class way to drive the model rather than something bolted on
    afterwards.

    Amplitudes are *relative* to each species' own steady value, so
    `amplitude = 1.0` means a species swings from zero to twice its steady
    concentration. Expressing them absolutely would make a modulation of trace
    acetylene invisible next to a modulation of bulk hydrogen.
    """
    base = constant_inlet(operating)

    def inlet(t: float) -> NDArray[np.float64]:
        value = base.copy()
        for frequency, amplitude in forcing or ():
            gain = np.cos(2.0 * np.pi * frequency * t)
            value += amplitude * base * gain
        return np.maximum(value, 0.0)

    return inlet


def steady_state_initial_condition(
    model: Model, *, ramp_time: float = 5.0, tolerance: float = 1e-10
) -> NDArray[np.float64]:
    """Initial state on the periodic-or-quasi-steady attractor.

    Starting a stiff transient from an empty reactor is a legitimate experiment
    and `zero_state` gives you exactly that. It is the wrong default for the
    question this thesis asks, because a surface that starts empty has to fill
    before it can respond to anything, and the filling is not what the
    measurement is about.

    The homotopy matters more than the tolerance. Switching the feed on
    discontinuously against a stiff source is exactly the initial-value problem
    an implicit integrator handles worst; ramping the inlet up over `ramp_time`
    and then integrating to convergence gets there reliably.

    The convergence test is on the *relative* change of the state over one
    further residence time. Using an absolute tolerance would either stop
    immediately for a trace species or never finish for a bulk one.
    """
    base = np.asarray(model.inlet(0.0), dtype=float)
    scale = float(np.max(np.abs(base))) or 1.0

    def ramp(t: float) -> NDArray[np.float64]:
        fraction = min(1.0, t / ramp_time) if ramp_time > 0 else 1.0
        # Smooth ramp: a linear one has a kink in its derivative, which the
        # stiff surface states feel as a spurious impulse.
        eased = fraction * fraction * (3.0 - 2.0 * fraction)
        return base * eased

    ramping = replace(model, inlet=ramp)
    y0 = np.zeros(model.layout.size)

    span = (0.0, ramp_time)
    first = ramping.integrate(y0, span, rtol=1e-8, atol=1e-14)
    if not first.success:
        raise RuntimeError(f"steady-state ramp failed: {first.message}")

    steady = first.y[:, -1]
    residence = model.derived.dz / model.derived.u_interstitial
    if residence <= 0.0:
        raise ValueError("non-positive residence time; check flow and geometry")

    # Keep integrating until nothing moves, with a hard cap so a pathological
    # parameter set fails loudly instead of hanging.
    span = (ramp_time, ramp_time + 200.0 * residence)
    for _ in range(12):
        result = model.integrate(steady, span, rtol=1e-9, atol=1e-15)
        if not result.success:
            raise RuntimeError(f"steady-state integration failed: {result.message}")
        change = np.max(np.abs(result.y[:, -1] - steady))
        steady = result.y[:, -1]
        if change <= tolerance * scale:
            return steady
    raise RuntimeError(
        "steady state did not converge in 12 continuation blocks; the rate law "
        "or the operating point is probably outside its fitted range"
    )


def zero_state(model: Model) -> NDArray[np.float64]:
    """Empty reactor: zero gas, zero coverage. A genuine step response."""
    return np.zeros(model.layout.size)


__all__ = [
    "Inputs",
    "build_model",
    "constant_inlet",
    "effective_diffusivities",
    "make_inlet",
    "steady_state_initial_condition",
    "zero_state",
]
