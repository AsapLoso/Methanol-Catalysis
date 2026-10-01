"""Parameter schemas.

Every quantity the model consumes is declared here, once, as a pydantic
model. Nothing downstream is allowed to carry a bare float that did not come
out of one of these: that is what makes the model auditable from the outside,
and it is what lets `tests/test_parameters.py` walk the registry and fail on
a unit nobody checked.

Three groups:

* `Geometry`, `Catalyst`, `OperatingPoint`, `TransportProperties` -- physics.
* `Larsen2024`, `Borodzinski2000`, `Urmes2019` -- one per kinetic literature
  source, so switching parameter sets is a type change rather than a set of
  magic numbers scattered through the code.
* `SolverSettings` -- discretisation and integrator choices.

Unit discipline: SI throughout (m, s, kg, mol, K, Pa), and partial pressures
in Pa. The sources disagree -- Borodzinski prints kPa, Larsen prints bar,
Urmes prints Pa -- so each kinetic set converts on the way in and the
conversion factor is part of that set's declaration rather than something a
call site has to remember.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

# CODATA / IUPAC. Not a "parameter" in the modelling sense; fixed physics.
R_GAS = 8.314462618  # J/(mol K)
M_PD = 106.42e-3  # kg/mol, molar mass of palladium
FARADAY = 96485.33212  # C/mol

#: Balance gases that carry no hydrogenation chemistry. Ar and He appear
#: across the source literature; N2 is accepted because it is common in this
#: literature and refusing it would be pedantry. An inert is excluded from the
#: reactive concentrations and never appears in a rate expression.
INERTS: frozenset[str] = frozenset({"Ar", "He", "N2"})


class AdvectionScheme(StrEnum):
    """First-order upwind, or second-order total-variation-diminished."""

    UPWIND1 = "upwind1"
    """Monotone, `O(dz)`. Diffusive. The safe default."""

    TVD2 = "tvd2"
    """SMCKF flux-limited, second order where the solution is smooth.

    Still monotone, so it cannot manufacture a negative concentration -- which
    matters here because the stiff source term would amplify any overshoot
    into visible ringing.
    """


class Geometry(BaseModel):
    """Reactor and bed geometry."""

    model_config = ConfigDict(frozen=True)

    tube_inner_diameter: float = Field(gt=0, description="m")
    catalyst_mass: float = Field(gt=0, description="kg of catalyst in the bed")
    bed_void_fraction: float = Field(
        default=0.45, gt=0.0, lt=1.0, description="eps_b, dimensionless"
    )

    @property
    def area(self) -> float:
        """Bed cross-section [m^2]."""
        return float(np.pi * self.tube_inner_diameter**2 / 4.0)

    def bed_length(self, catalyst_bulk_density: float) -> float:
        """Packed height [m] for a given bulk catalyst density.

        Derived rather than stored because the bulk density belongs to the
        catalyst, not the tube. Note this is `W / (rho_bulk * A)` with
        `rho_bulk = rho_particle * (1 - eps_b)`; the benchmark thesis used
        `W / (rho_support * A)` and so under-reports the bed height by a
        factor `(1 - eps_b)`. We keep the void fraction explicit.
        """
        if catalyst_bulk_density <= 0.0:
            raise ValueError("catalyst_bulk_density must be positive")
        return self.catalyst_mass / (catalyst_bulk_density * self.area)


class Catalyst(BaseModel):
    """Pellet properties. These set every transport timescale."""

    model_config = ConfigDict(frozen=True)

    name: str = "unnamed"
    support: str = "unspecified"
    particle_radius: float = Field(gt=0, description="R_p [m]")
    particle_porosity: float = Field(default=0.5, gt=0.0, le=1.0, description="eps_p")
    particle_density: float = Field(gt=0, description="rho_p [kg/m^3]")
    tortuosity: float = Field(default=1.0, ge=1.0)
    metal_mass_fraction: float = Field(
        default=0.0, ge=0.0, le=1.0, description="kg metal per kg catalyst"
    )
    site_density: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Active sites per kg catalyst [mol_site/kg_cat]. Zero means the "
            "rate laws are already normalised per kg catalyst and must not "
            "be converted again."
        ),
    )

    @property
    def metal_site_density(self) -> float:
        """Metal sites per kg catalyst [mol_M/kg_cat], from mass loading.

        Used by rate laws published per mole of metal (Borodzinski, Urmes).
        Returns 0.0 when the catalyst declares no metal, so the caller gets a
        loud divide-by-nothing rather than a silent wrong answer.
        """
        if self.metal_mass_fraction <= 0.0:
            return 0.0
        return self.metal_mass_fraction / M_PD


class OperatingPoint(BaseModel):
    """Feed and thermodynamic state."""

    model_config = ConfigDict(frozen=True)

    temperature: float = Field(default=373.15, gt=0, description="K, isothermal")
    pressure: float = Field(default=101325.0, gt=0, description="Pa")
    volumetric_flow: float = Field(gt=0, description="total feed Q [m^3/s]")
    mole_fractions: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_composition(self) -> OperatingPoint:
        allowed = {"H2", "C2H2", "C2H4", "C2H6"} | INERTS
        unknown = set(self.mole_fractions) - allowed
        if unknown:
            raise ValueError(
                f"unknown feed species {sorted(unknown)}; allowed: {sorted(allowed)}"
            )
        for name, y in self.mole_fractions.items():
            if not 0.0 <= y <= 1.0:
                raise ValueError(f"mole fraction of {name} is {y}, outside [0, 1]")
        return self

    @property
    def inert(self) -> str | None:
        """Which inert balance gas, if any. `None` for an inert-free feed."""
        for name in self.mole_fractions:
            if name in INERTS:
                return name
        return None

    @property
    def inerts_present(self) -> bool:
        return self.inert is not None

    @property
    def reacting_mole_fraction(self) -> float:
        return sum(y for name, y in self.mole_fractions.items() if name not in INERTS)

    def concentrations(self) -> dict[str, float]:
        """Gas-phase concentration per unit *void* volume [mol/m^3].

        `y_i P / (RT)` gives mol per m^3 of the whole stream. With an inert
        present that is not the void concentration, so the reactive species are
        rescaled by the reacting fraction. With no inert the two coincide and
        the division is skipped rather than left in as a no-op.

        The inert is excluded from the output entirely: it is not on the
        hydrogenation network and carrying it would mean inventing a fourth
        rate for it.
        """
        reacting = self.reacting_mole_fraction
        rt = R_GAS * self.temperature
        scale = self.pressure if reacting == 0.0 else self.pressure / reacting
        return {
            name: y * scale / rt
            for name, y in self.mole_fractions.items()
            if name not in INERTS
        }

    def partial_pressures(self) -> dict[str, float]:
        """Partial pressure per unit void volume [Pa], consistent with above."""
        c = self.concentrations()
        return {name: value * R_GAS * self.temperature for name, value in c.items()}


class TransportProperties(BaseModel):
    """Effective transport coefficients.

    `effective_diffusivity` is *porosity-inclusive* (`D_e = eps_p * D_m / tau`),
    matching the definition used by the transport reference. The intraparticle
    balance therefore divides by `eps_p` explicitly. The alternative
    convention -- an `eps_p`-exclusive `D_e` -- changes the result by `eps_p`
    and is the single most common source of silent factor-of-two errors in
    pellet models, so the convention is fixed here and stated in the docs.
    """

    model_config = ConfigDict(frozen=True)

    effective_diffusivity: dict[str, float] = Field(default_factory=dict)
    """Species -> D_e [m^2/s]. Missing species fall back to a supplied default."""

    default_diffusivity: float = Field(default=1.0e-6, gt=0.0)

    def diffusivity(self, species: str) -> float:
        return self.effective_diffusivity.get(species, self.default_diffusivity)


class SolverSettings(BaseModel):
    """Discretisation and integrator."""

    model_config = ConfigDict(frozen=True)

    n_axial: int = Field(default=60, ge=2, description="N_z, cell count")
    n_radial: int = Field(default=25, ge=2, description="N_r, radial nodes")
    """Odd by design. The particle-volume quadrature uses composite Simpson,
    which needs an even number of intervals and therefore an odd node count.
    An even count silently falls back to trapezoid and reinstates an `O(dr^2)`
    bias in every particle-averaged rate."""
    radial_geometry: int = Field(default=2, ge=0, le=2, description="s")
    """`s = 2` sphere, `1` cylinder, `0` slab. Set from the pellet shape."""

    advection_scheme: AdvectionScheme = AdvectionScheme.TVD2
    axial_dispersion: float = Field(
        default=0.0, ge=0.0, description="D_ax [m^2/s]; 0 means plug flow"
    )
    integrator: str = Field(default="BDF")
    rtol: float = Field(default=1e-7, gt=0)
    atol: float = Field(default=1e-12, gt=0)
    max_step: float | None = Field(default=None, gt=0)
    use_sparsity: bool = Field(
        default=True,
        description=(
            "Supply the analytic Jacobian sparsity pattern to solve_ivp. The "
            "dense finite-difference Jacobian costs one full rhs per state, "
            "which for a resolved pellet is thousands of them."
        ),
    )

    @model_validator(mode="after")
    def _scheme_needs_dispersion(self) -> SolverSettings:
        if self.axial_dispersion > 0.0:
            # A centred second-order stencil is only stable with dispersion to
            # damp it; a pure hyperbolic upwind stencil needs no second order.
            object.__setattr__(self, "advection_scheme", AdvectionScheme.UPWIND1)
        return self

    @property
    def dispersion_coefficient(self) -> float:
        """`D_ax` in m^2/s, allowing for the Edwards-Richardson fallback.

        With no value supplied, zero -- plug flow. That is a decision, not a
        default: the axial operator then stays hyperbolic and the outflow
        boundary condition is zero-gradient rather than Danckwerts.
        """
        return self.axial_dispersion


# --- Kinetic parameter sets ------------------------------------------------


class KineticSet(BaseModel):
    """Base for a literature parameterisation.

    Subclasses fill `k` (rate constants in the source's own pressure unit) and
    `K` (adsorption constants, inverse pressure in the same unit). The
    `pressure_scale` converts the source's pressure unit to Pa, so the
    expressions in `rate_laws.py` can be written once in Pa.
    """

    model_config = ConfigDict(frozen=True)

    reference: str = Field(description="citation key into references.bib")
    pressure_scale: float = Field(
        description="Pa per unit of the source's pressure unit (1e5 for bar, 1e3 for kPa)"
    )
    temperature: float = Field(gt=0, description="T the set was fitted at [K]")
    mass_scale: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Source mass basis to kg catalyst. 1e3 where rates are published "
            "per gram, 1.0 where per kilogram."
        ),
    )
    site_basis: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Factor taking the source's site normalisation to mol per kg "
            "catalyst. 1.0 where rates are already per kg catalyst; otherwise "
            "the metal site density [mol_M/kg_cat]. Leaving this at 1.0 by "
            "accident is a silent 10^3 error, which is why it is a field "
            "rather than something a rate law infers."
        ),
    )

    @property
    def orders(self) -> dict[str, int]:
        """How many *bare* pressure factors each rate constant carries.

        Read this off the source's declared units for `k`, and nothing else
        decides it:

        * Borodzinski prints `k1 = 12.63 mol / mol_Pd / s / kPa^-3`. The `kPa^-3`
          is in the units, so `k1` carries three pressure factors and must be
          divided by `pressure_scale ** 3` to be used against Pa.
        * Larsen prints `k1 = 7.1852e-5 mol/(g_cat s)` -- no pressure factor at
          all. All of its pressure dependence arrives through explicit `K * p`
          products in the numerator, which are already invariant under the unit
          change. Dividing its `k` as well would rescale every rate by `1e10`.

        So the count is per parameter set, not a property of the model, and
        guessing it from the apparent reaction order gets it wrong. The reaction
        order counts *all* pressure factors; this counts only the ones the
        source baked into `k`. Getting that distinction wrong is a silent
        factor of `(1e5)^n` on every rate in the model.
        """
        raise NotImplementedError

    def rate_constant(self, name: str) -> float:
        """`k` in mol/(kg_cat s), valid against partial pressures in Pa.

        Three conversions: the source's mass basis to kg catalyst, its site
        normalisation to mol per kg catalyst, and its pressure basis to Pa.
        The last one is `pressure_scale ** orders[name]`, which is 1 for any set
        whose `k` carries no pressure factor in its units.
        """
        return (
            self.k[name]
            * self.mass_scale
            * self.site_basis
            / self.pressure_scale ** self.orders[name]
        )

    def adsorption_constant(self, name: str) -> float:
        """`K` in Pa^-1 (or Pa for a desorption constant), for pressures in Pa."""
        return self.K[name] / self.pressure_scale


class Larsen2024(KineticSet):
    """Extended LHHW with a saturating primary rate and flexible inhibition.

    Two departures from textbook LHHW, both empirical and both load-bearing:

    `r_1 = r_raw / (1 + r_raw / r_1max)` caps the primary reaction. The
    source introduced it because plain LHHW compensates between `k_1` and
    `K_C2H2` so hard that it predicts ~100% conversion everywhere; the cap is a
    site-availability ceiling standing in for what the fitted parameters cannot
    express.

    `(1 + K_inh * p_C2H2) ** n_inh` on `r_2` describes acetylene poisoning the
    ethylene-hydrogenation site. `n_inh` is an exponent, not a constant, and
    `1.0` reproduces the source's linear form.
    """

    reference: str = "larsenSemihydrogenationAcetyleneEthylene2024"
    pressure_scale: float = 1.0e5  # bar -> Pa
    temperature: float = 373.15

    k1: float = Field(gt=0, description="mol/(g_cat s) before scaling")
    k2: float = Field(gt=0, description="mol/(g_cat s) before scaling")
    r_1max: float = Field(gt=0, description="mol/(g_cat s)")

    K_C2H2: float = Field(gt=0)
    K_C2H4: float = Field(gt=0)
    K_H2: float = Field(gt=0)
    K_inh: float = Field(ge=0)
    n_inh: float = Field(default=1.0, ge=0.0)

    #: Rates published per gram of catalyst; the model works per kilogram.
    mass_scale: float = 1.0e3

    @property
    def k(self) -> dict[str, float]:
        return {"k1": self.k1, "k2": self.k2}

    @property
    def K(self) -> dict[str, float]:
        return {
            "C2H2": self.K_C2H2,
            "C2H4": self.K_C2H4,
            "H2": self.K_H2,
            "inh": self.K_inh,
        }

    @property
    def orders(self) -> dict[str, int]:
        # `k1` is published as mol/(g_cat s) with no pressure factor, and all
        # pressure dependence enters through explicit `K * p` products.
        return {"k1": 0, "k2": 0}


class Borodzinski2000(KineticSet):
    """Two-site Langmuir-Hinshelwood model on carbon-covered palladium.

    Carbonaceous deposits split the surface into narrow `A` sites, which
    acetylene and hydrogen can reach but ethylene cannot, and wide `E` sites,
    where all three compete. That geometric exclusion is the mechanism behind
    the high ethylene selectivity, and it is why the model has two separate
    `K_C2H2` values and no `A`-site constant for ethylene at all.

    Rates are published per mole of *palladium*, so each `k` must be multiplied
    by the catalyst's metal site density to reach mol/kg catalyst. That factor
    is not optional and not part of `k`: it belongs to the catalyst.
    """

    reference: str = "borodzinskiKineticModelHydrogenation2000"
    pressure_scale: float = 1.0e3  # kPa -> Pa
    temperature: float = 343.0

    k1: float = Field(gt=0, description="per mol Pd, 3rd order in pressure")
    k2: float = Field(gt=0, description="per mol Pd, 2nd order")
    k3: float = Field(gt=0, description="per mol Pd, 2nd order")
    k4: float = Field(gt=0, description="per mol Pd, 2nd order")
    k5: float = Field(gt=0, description="per mol Pd, 3rd order")

    K_C2H2_A: float = Field(gt=0, description="acetylene on A sites")
    K_C2H2_E: float = Field(gt=0, description="acetylene on E sites")
    K_C2H4_E: float = Field(gt=0, description="ethylene on E sites")

    @property
    def k(self) -> dict[str, float]:
        return {f"k{i}": getattr(self, f"k{i}") for i in range(1, 6)}

    @property
    def K(self) -> dict[str, float]:
        return {
            "C2H2_A": self.K_C2H2_A,
            "C2H2_E": self.K_C2H2_E,
            "C2H4_E": self.K_C2H4_E,
        }

    @property
    def orders(self) -> dict[str, int]:
        return {"k1": 3, "k2": 2, "k3": 2, "k4": 2, "k5": 3}


class Urmes2019(KineticSet):
    """Single-site Horiuti-Polanyi sequential hydrogen addition.

    This is the only set in the database that publishes *activation energies*
    as well as pre-exponentials, which makes it the only one that can say
    anything about temperature. It is therefore the default for anything
    thermal, and the reason `SolverSettings` carries no temperature dependence
    at all when the other two sets are selected.

    Dissociative adsorption puts `sqrt(K_H2 p_H2)` rather than `K_H2 p_H2` in
    the denominator, and ethylene desorbs into the denominator through
    `p_C2H4 / K_C2H4_des`.
    """

    reference: str = "urmesKineticStudySelective2019"
    pressure_scale: float = 1.0  # already Pa
    temperature: float = 324.15

    k04: float = Field(gt=0, description="pre-exponential, acetylene RDS")
    k07: float = Field(gt=0, description="pre-exponential, ethylene RDS")
    E4: float = Field(gt=0, description="kJ/mol")
    E7: float = Field(gt=0, description="kJ/mol")

    K_H2: float = Field(gt=0, description="dissociative H2, Pa^-1")
    K_C2H2: float = Field(gt=0, description="Pa^-1")
    K_C2H4_des: float = Field(gt=0, description="desorption constant, Pa")

    site_density: float = Field(
        gt=0, description="N_S [mol Pd_s/kg_cat], the normalisation of k04/k07"
    )

    @property
    def k(self) -> dict[str, float]:
        return {"k04": self.k04, "k07": self.k07}

    @property
    def orders(self) -> dict[str, int]:
        return {"k04": 0, "k07": 0}

    def arrhenius(self, name: str, temperature: float) -> float:
        """`k04` or `k07` at `temperature`, in mol/(mol_Pd_s s).

        Plain `A exp(-E / RT)`, **not** the centred form.

        This is worth being explicit about, because the centred form
        (`k_ref exp(-E/R (1/T - 1/T_ref))`) is the right choice elsewhere in
        this literature -- it is what the dynamic-kinetics reference uses to
        decorrelate `A` from `E` -- and reaching for it here by habit is
        exactly the wrong call. The centred form assumes `k04` is the rate
        constant *at* `T_ref`; this source publishes it as a pre-exponential
        and its own rate equation reads `k04 exp(-E4/RT)`. Applying the
        centred form to a genuine pre-exponential multiplies every rate by
        `exp(E4 / (R T_ref))` -- a factor of `6.4e7` for these numbers, and
        entirely silent, because it is a constant and constants do not look
        wrong.

        The source's own equation decides it. A centred parameterisation is a
        *re-parameterisation* you apply to a fitted rate constant; it is not
        something you apply to an already-decomposed `A` and `E`.
        """
        Ea = self.E4 if name == "k04" else self.E7
        prefactor = self.k04 if name == "k04" else self.k07
        return float(prefactor * np.exp(-(Ea * 1.0e3 / R_GAS) / temperature))


#: Every parameterisation the model knows how to evaluate, by name. Populated
#: in `pbr.parameter_sets`; kept here so a caller can enumerate them without
#: importing the concrete sets, which is what the CLI and the docs generator do.
KINETIC_SETS: dict[str, str] = {
    "urmes2019": "single-site Horiuti-Polanyi, with activation energies",
    "borodzinski2000": "two-site LHHW, A/E site populations",
    "larsen2024": "extended LHHW, rate saturation + acetylene inhibition",
}


def fuller_diffusivity(
    temperature: float,
    pressure: float,
    mass_a: float,
    mass_b: float,
    vol_a: float,
    vol_b: float,
) -> float:
    """Gas-phase binary diffusivity [m^2/s] for a polyatomic gas pair.

    Fuller-Schettler-Giddings. `vol_a`, `vol_b` are diffusion volumes, tabulated
    in the source; pass them in rather than looking them up here, so a call
    site states which pair it means.

    Used to build `D_e = eps_p D_m / tau`. That is a *derived* value with a
    correlation behind it, which is strictly better than a flat placeholder --
    but it is still a correlation for a different geometry than a real porous
    particle, and `docs/worklog.md` sweeps the tortuosity that stands in for the
    pore structure.
    """
    d_cm2_s = (
        1.0e-2
        * 1.43e-1
        * temperature**1.75
        / (
            pressure
            * (2.0 / (1.0 / mass_a + 1.0 / mass_b)) ** 0.5
            * (vol_a ** (1.0 / 3.0) + vol_b ** (1.0 / 3.0)) ** 2
        )
    )
    return d_cm2_s * 1.0e-4


#: Diffusion volumes for the gas pairs this model needs [dimensionless].
#: Fuller-Schettler-Giddins tabulation.
DIFFUSION_VOLUMES: dict[str, float] = {
    "H2": 7.07,
    "Ar": 16.1,
    "He": 6.12,
    "C2H2": 37.8,
    "C2H4": 41.6,
    "C2H6": 52.5,
}

#: Molar masses for the Fuller correlation [g/mol].
MASSES: dict[str, float] = {
    "H2": 2.016,
    "Ar": 39.948,
    "He": 4.003,
    "C2H2": 26.038,
    "C2H4": 28.054,
    "C2H6": 30.070,
}
