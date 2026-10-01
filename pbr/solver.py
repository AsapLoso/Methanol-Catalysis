"""The method-of-lines right-hand side, and the integrator that runs it.

Balances, following `bergerDynamicMethodsCatalytic2008` Eq. (1) and Eq. (2),
on a void-volume basis:

    dC_b,i/dt = -(u_s/eps_b) dC_b,i/dz
                + (D_ax/eps_b) d2C_b,i/dz2
                + (rho_s (1-eps_b)/eps_b) <R_i>_particle

    dc_p,i/dt = (D_e/eps_p) (1/r^s) d/dr (r^s dc_p,i/dr) + (rho_s/eps_p) R_i

    dtheta_j/dt = (theta_j^ss - theta_j)/tau

Three conventions that had to be pinned down. Each is a silent factor of
`eps_p` or `eps_b` if you get it wrong, and each is stated here rather than
left implicit:

`u_s` is the **superficial** velocity, based on the empty tube cross-section.
The interstitial velocity is `u_s / eps_b`, and the interstitial gas is what
the advection term transports, so the transport term carries `u_s / eps_b`.
(The working notes in the BEP tree write a bare `u_s` against a bulk-volume
concentration. That is the same physics with the void fraction folded into the
concentration instead -- not a contradiction, but the two forms cannot be mixed
without introducing exactly one spurious `eps_b`.)

`D_e` is **porosity-inclusive**, `D_e = eps_p D_m / tau`, matching the
transport reference, and is therefore divided by `eps_p` in the pellet balance
to give diffusivity per unit pore volume. Had it been defined as `D_m / tau`,
that division would have to disappear. Both conventions are in circulation and
they differ by exactly `eps_p`.

The rate law is normalised to mol/(kg_cat s), so the geometric prefactor
carries the density: `rho_s / eps_p` converts it to mol/(m^3_pore s) inside the
pellet, and `rho_s (1 - eps_b) / eps_b` to mol/(m^3_void s) in the bulk. The
two are consistent by construction -- the particle equation volume-averaged
over the pellet and multiplied by the bed solid fraction reproduces the bulk
term exactly -- which `tests/test_solver.py::test_particle_average_reproduces_bulk_source`
checks numerically rather than trusting.

Evaluation order is fixed and matters: bulk, shell, particle, chemistry, then
the bulk source from the volume-averaged rate. `C_s` depends on `C_b`; the rate
depends on `c_p`; the bulk source depends on the volume average of the rate.
Any other order consumes a value from a previous stage, which on a stiff
problem is the difference between a converging step and a diverging one.

**Nothing here writes to its input.** The state vector is unpacked into fresh
arrays before any boundary condition is applied. An earlier version of this
model assigned the shell condition through a `reshape` *view* of the solver's
own state vector, so every rhs call silently overwrote 80 of 1480 entries of
the trajectory the integrator was storing -- the stored trajectory was not the
trajectory the equations describe.
`tests/test_solver.py::test_rhs_does_not_mutate_its_input` pins that down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

from pbr.chemistry import RateLaw
from pbr.closures import (
    AxialDispersion,
    ExternalTransport,
    RadialTransport,
    SurfaceDynamics,
)
from pbr.discretisation import (
    RadialGrid,
    axial_advection,
    axial_dispersion,
    jacobian_sparsity,
    radial_diffusion,
)
from pbr.parameters import R_GAS
from pbr.species import GASES, IGAS, atom_counts

N_ADSORBATE_STATES = 3  # H*, C2H2*, C2H4*; the vacant site is 1 - sum.


class Inlet(Protocol):
    """`t -> (n_gas,)` inlet concentration in mol/m^3 of void gas."""

    def __call__(self, t: float) -> NDArray[np.float64]: ...


@dataclass(frozen=True, slots=True)
class Derived:
    """Quantities computed once from the inputs, so the hot loop is arithmetic.

    A geometry error should surface when the model is built, not as a NaN three
    thousand derivatives into an integration.
    """

    u_superficial: float
    u_interstitial: float
    bed_length: float
    dz: float
    n_z: int
    particle_source_factor: float
    """rho_s / eps_p [kg_cat/m^3_pore]."""
    bulk_source_factor: float
    """rho_s (1 - eps_b) / eps_b [kg_cat/m^3_void]."""
    particle_porosity: float
    void_fraction: float


def derive(
    *,
    n_z: int,
    tube_inner_diameter: float,
    catalyst_mass: float,
    void_fraction: float,
    particle_density: float,
    particle_porosity: float,
    volumetric_flow: float,
) -> Derived:
    """Turn the raw parameter set into the numbers `rhs` actually uses."""
    if n_z < 2:
        raise ValueError("need at least two axial cells")
    if not 0.0 < void_fraction < 1.0:
        raise ValueError(f"void fraction {void_fraction} outside (0, 1)")
    if particle_porosity <= 0.0:
        raise ValueError("particle porosity must be positive")

    area = float(np.pi * tube_inner_diameter**2 / 4.0)
    bulk_density = particle_density * (1.0 - void_fraction)
    bed_length = catalyst_mass / (bulk_density * area)
    u_superficial = volumetric_flow / area

    return Derived(
        u_superficial=u_superficial,
        u_interstitial=u_superficial / void_fraction,
        bed_length=bed_length,
        dz=bed_length / n_z,
        n_z=n_z,
        particle_source_factor=particle_density / particle_porosity,
        bulk_source_factor=bulk_density / void_fraction,
        particle_porosity=particle_porosity,
        void_fraction=void_fraction,
    )


@dataclass(frozen=True, slots=True)
class Layout:
    """Where each field lives in the flat state vector.

    Slicing and copying, never a view. The view was the original defect.
    """

    n_z: int
    n_gas: int
    n_radial: int
    carries_radial: bool
    carries_surface: bool

    @property
    def n_radial_effective(self) -> int:
        return self.n_radial if self.carries_radial else 1

    @property
    def gas_size(self) -> int:
        return self.n_z * self.n_gas

    @property
    def radial_size(self) -> int:
        return self.gas_size * self.n_radial if self.carries_radial else 0

    @property
    def surface_size(self) -> int:
        if not self.carries_surface:
            return 0
        return self.n_z * self.n_radial_effective * N_ADSORBATE_STATES

    @property
    def size(self) -> int:
        return self.gas_size + self.radial_size + self.surface_size

    def pack(
        self,
        c_bulk: NDArray[np.float64],
        c_particle: NDArray[np.float64] | None = None,
        theta: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Flatten the fields. `c_particle`/`theta` ignored unless carried."""
        parts = [np.ravel(np.asarray(c_bulk, dtype=float))]
        if self.carries_radial:
            if c_particle is None:
                raise ValueError("radial closure carries states but none supplied")
            parts.append(np.ravel(np.asarray(c_particle, dtype=float)))
        if self.carries_surface:
            if theta is None:
                raise ValueError("surface closure carries states but none supplied")
            parts.append(np.ravel(np.asarray(theta, dtype=float)))
        out = np.concatenate(parts)
        if out.size != self.size:
            raise ValueError(f"packed {out.size} values, layout expects {self.size}")
        return out

    def unpack(self, y: NDArray[np.float64]) -> tuple:
        """Split into freshly allocated `(c_bulk, c_particle, theta)`.

        Every returned array is a copy, so a caller may write to it freely.
        """
        y = np.asarray(y, dtype=float)
        c_bulk = y[: self.gas_size].reshape(self.n_z, self.n_gas).copy()

        c_particle = None
        theta = None
        offset = self.gas_size
        if self.carries_radial:
            stop = offset + self.radial_size
            c_particle = (
                y[offset:stop].reshape(self.n_z, self.n_radial, self.n_gas).copy()
            )
            offset = stop
        if self.carries_surface:
            stop = offset + self.surface_size
            theta = (
                y[offset:stop]
                .reshape(self.n_z, self.n_radial_effective, N_ADSORBATE_STATES)
                .copy()
            )
        return c_bulk, c_particle, theta


@dataclass(slots=True)
class Model:
    """A fully specified model. Everything `rhs` needs, and nothing it does not."""

    layout: Layout
    radial: RadialTransport
    external: ExternalTransport
    surface: SurfaceDynamics
    dispersion: AxialDispersion
    rate: RateLaw
    grid: RadialGrid
    derived: Derived
    temperature: float
    diffusivities: dict[str, float]
    advection_scheme: str
    inlet: Inlet
    _sparsity: csr_matrix | None = field(default=None, init=False, repr=False)
    _d_eff: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._d_eff = np.array(
            [self.diffusivities[name] for name in GASES], dtype=float
        )

    # --- evaluation --------------------------------------------------------

    def shell_concentration(
        self,
        c_bulk: NDArray[np.float64],
        c_particle: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """`c_p` at `r = R_p` for every species and axial cell."""
        c_inner = c_particle[:, -1, :]
        out = np.empty_like(c_bulk)
        for j, _ in enumerate(GASES):
            out[:, j] = self.external.shell_concentration(
                c_bulk[:, j],
                c_inner[:, j],
                float(self._d_eff[j]),
                self.grid.dr,
            )
        return out

    def rhs(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """`dy/dt`. Pure in `y`."""
        c_bulk, c_particle, theta = self.layout.unpack(y)
        rt = R_GAS * self.temperature
        dz = self.derived.dz
        c_in = np.asarray(self.inlet(t), dtype=float)

        # --- particle field and the rate law ------------------------------
        if self.layout.carries_radial:
            shell = self.shell_concentration(c_bulk, c_particle)
            local_p = c_particle * rt
            result = self.rate.rates(local_p, self.temperature)
            weights = self.grid.weights
            mean_rate = np.einsum("r,zri->zi", weights, result.gas)
        else:
            shell = c_bulk
            result = self.rate.rates(c_bulk * rt, self.temperature)
            mean_rate = result.gas

        # --- bulk: transport, then source ---------------------------------
        # Both transport terms are divided by eps_b because C_b is a
        # void-volume concentration while u_s is a cross-section-average
        # velocity and D_ax is defined per unit total volume.
        d_c_bulk = np.empty_like(c_bulk)
        d_ax = self.dispersion.coefficient / self.derived.void_fraction
        for j in range(self.layout.n_gas):
            column = c_bulk[:, j : j + 1]
            advection = axial_advection(
                column,
                self.derived.u_interstitial,
                dz,
                c_in[j : j + 1],
                self.advection_scheme,
            )
            dispersion = axial_dispersion(column, d_ax, dz)
            d_c_bulk[:, j] = advection[:, 0] + dispersion[:, 0]
        d_c_bulk += self.derived.bulk_source_factor * mean_rate

        # --- particle ------------------------------------------------------
        d_c_particle = None
        if self.layout.carries_radial:
            d_c_particle = np.empty_like(c_particle)
            for j in range(self.layout.n_gas):
                d_c_particle[:, :, j] = (
                    radial_diffusion(
                        c_particle[:, :, j],
                        self.grid,
                        float(self._d_eff[j]),
                        self.derived.particle_porosity,
                        shell=shell[:, j : j + 1],
                    )
                    + self.derived.particle_source_factor * result.gas[:, :, j]
                )

        # --- surface -------------------------------------------------------
        d_theta = None
        if self.layout.carries_surface:
            target = self.rate.coverages(c_bulk * rt, self.temperature)
            if target is None:
                raise NotImplementedError(
                    f"rate law {self.rate.name!r} does not expose coverages, so "
                    f"it cannot drive a relaxing surface. Either choose a "
                    f"rate law that does (Urmes2019) or use the algebraic "
                    f"surface closure."
                )
            d_theta = self.surface.derivative(theta, target)

        return self.layout.pack(d_c_bulk, d_c_particle, d_theta)

    # --- integration -------------------------------------------------------

    @property
    def sparsity(self) -> csr_matrix:
        """Analytic sparsity pattern, built once and cached."""
        if self._sparsity is None:
            self._sparsity = jacobian_sparsity(
                self.layout.n_z,
                self.layout.n_gas,
                self.layout.n_radial if self.layout.carries_radial else None,
                (
                    self.layout.n_radial_effective
                    if self.layout.carries_surface
                    else None
                ),
            )
        return self._sparsity

    def integrate(
        self,
        y0: NDArray[np.float64],
        t_span: tuple[float, float],
        *,
        method: str = "BDF",
        rtol: float = 1e-7,
        atol: float = 1e-12,
        max_step: float | None = None,
        use_sparsity: bool = True,
        dense_output: bool = False,
    ):
        """Integrate with an implicit method and, by default, a sparse Jacobian.

        The Jacobian is the cost, not the right-hand side. SciPy estimates it by
        perturbing each state variable in turn, so a dense Jacobian costs one
        full rhs per unknown -- thousands of evaluations for a resolved pellet,
        on every step that needs a fresh one. The analytic sparsity pattern
        reduces that to a small number of colour groups instead, which is the
        single highest-leverage line in this module.
        """
        kwargs: dict = {
            "method": method,
            "rtol": rtol,
            "atol": atol,
            "dense_output": dense_output,
        }
        if use_sparsity:
            kwargs["jac_sparsity"] = self.sparsity
        if max_step is not None:
            kwargs["max_step"] = max_step
        return solve_ivp(self.rhs, t_span, y0, **kwargs)


def atom_balance(y: NDArray[np.float64], layout: Layout) -> tuple[float, ...]:
    """Carbon and hydrogen totals implied by a state vector.

    A cheap invariant. A stoichiometry table transcribed wrong by one shows up
    here as an imbalance that grows with reaction extent, rather than as a rate
    that merely looks odd.
    """
    c_bulk, _c_particle, theta = layout.unpack(np.asarray(y, dtype=float))
    n_gas = {name: float(c_bulk[:, IGAS[name]].sum()) for name in GASES}
    n_ads = {"H*": 0.0, "C2H2*": 0.0, "C2H4*": 0.0}
    if theta is not None:
        for index, name in enumerate(("H*", "C2H2*", "C2H4*")):
            n_ads[name] = float(theta[:, :, index].sum())
    return atom_counts(n_gas, n_ads)
