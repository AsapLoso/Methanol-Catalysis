"""Rate laws, one per literature parameterisation, behind one interface.

Every rate law here is *macro*: it returns a rate directly from partial
pressures and temperature, with no adsorbed intermediate appearing explicitly.
That is not laziness, it is what the available data supports. The parameter
database holds composite rate constants -- products of elementary constants
that the source chose to report as one number -- so there is no way to
reconstruct `dtheta/dt` from them. Refusing to invent the missing elementary
constants costs us mechanistic detail and buys us a model whose every rate is
traceable to a table in a published paper.

What that costs is the pseudo-steady-state assumption: a macro rate law has,
by construction, already assumed the surface is at equilibrium. `Urmes2019Rate`
therefore does something the other two cannot. Its adsorption terms are
reported individually rather than as products, so its coverages can be written
down, and with `Relaxation` surface dynamics it becomes a genuine transient
model. That is why it is the default.

Interface. `rates` takes partial pressures in Pa and returns

    gas[i]     mol/(kg_cat s)  net production of gas species i
    surface[j] 1/s             net production of coverage j (per site)

Both are *already* normalised to per kilogram of catalyst. Sources publish per
gram, per mole of metal, or per kilogram; the conversion lives in
`KineticSet.mass_scale` / `site_basis`, so no expression below carries a unit
conversion. `r_1max` is the one exception: it is a rate rather than a
pressure-dependent coefficient, so it is not divided by `pressure_scale`.

All three laws are monotone in hydrogen and *decrease* in acetylene over part
of their range -- the rate maximum the sources both report. Preserving that
non-monotonicity is the point: a model in which the rate rises monotonically
in its own reactant cannot exhibit the competitive-adsorption regime that
makes periodic operation worth studying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from pbr.parameters import (
    Borodzinski2000,
    KineticSet,
    Larsen2024,
    Urmes2019,
)
from pbr.species import GASES, IGAS


@dataclass(frozen=True, slots=True)
class RateResult:
    """Volumetric rates out of the rate law.

    `gas` and `surface` are stacked rather than returned separately so a caller
    cannot get one without the other -- they are never independently meaningful.
    """

    gas: NDArray[np.float64]
    """Shape (..., 4) in the order of `pbr.species.GASES`. mol/(kg_cat s)."""

    surface: NDArray[np.float64]
    """Shape (..., 3) for H*, C2H2*, C2H4*. 1/s. All zeros if unavailable."""

    @property
    def has_surface(self) -> bool:
        return bool(np.any(self.surface))


@runtime_checkable
class RateLaw(Protocol):
    """A rate law. The only thing `rhs` needs to know about chemistry."""

    @property
    def name(self) -> str: ...

    def rates(
        self,
        pressures: NDArray[np.float64],
        temperature: float,
    ) -> RateResult:
        """Evaluate at partial pressures in Pa.

        `pressures` broadcasts over leading axes, so a (N_z, N_r) stack of
        local pressures gives a (N_z, N_r, 4) result without a Python loop.
        """

    def coverages(
        self,
        pressures: NDArray[np.float64],
        temperature: float,
    ) -> NDArray[np.float64] | None:
        """Quasi-steady adsorbed fractions in the order H*, C2H2*, C2H4*.

        `None` when the parameter set does not permit recovery. The two-site
        model is the case that cannot: its constants are products of
        elementary constants, so the individual site populations are not
        identifiable from the published numbers. Returning `None` rather than
        an approximate answer is deliberate.
        """


def _pressures(p: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    """Pull the four reactive partial pressures out of an array."""
    return {name: p[..., IGAS[name]] for name in GASES if name != "H2"} | {
        "H2": p[..., IGAS["H2"]]
    }


def _gas_stack(
    h2: NDArray[np.float64],
    c2h2: NDArray[np.float64],
    c2h4: NDArray[np.float64],
    c2h6: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.stack([h2, c2h2, c2h4, c2h6], axis=-1)


# --- Larsen 2024 ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Larsen2024Rate:
    """Extended LHHW with a saturating primary rate.

    The two empirical pieces are kept verbatim because they are the reason this
    model exists. Plain LHHW cannot fit these experiments: `k1` and `K_C2H2`
    trade off against each other so completely that the fitted pair predicts
    total conversion at every operating point, which is a curve that carries
    no information. The saturation cap `r_1max` and the free inhibition
    exponent `n_inh` are what make the fit mean something.

    The cap is written as a Michaelis-Menten saturation rather than a hard
    `min()`, because a hard clip is discontinuous in `p_C2H2` and would make
    the integrator take vanishing steps every time the rate crosses it.
    """

    params: Larsen2024

    @property
    def name(self) -> str:
        return "Larsen2024 (extended LHHW, rate saturation + acetylene inhibition)"

    @property
    def _k1(self) -> float:
        return self.params.rate_constant("k1")

    @property
    def _k2(self) -> float:
        return self.params.rate_constant("k2")

    @property
    def _r_cap(self) -> float:
        """The cap, on the model's own per-kilogram basis.

        Not divided by `pressure_scale`: it is a rate, not a coefficient with
        pressure factors in it.
        """
        return self.params.r_1max * self.params.mass_scale * self.params.site_basis

    def rates(
        self,
        pressures: NDArray[np.float64],
        temperature: float,
    ) -> RateResult:
        p = pressures
        ph2 = p[..., IGAS["H2"]]
        pa = p[..., IGAS["C2H2"]]
        pb = p[..., IGAS["C2H4"]]

        k_acetylene = self.params.adsorption_constant("C2H2")
        k_ethylene = self.params.adsorption_constant("C2H4")
        k_hydrogen = self.params.adsorption_constant("H2")
        k_inh = self.params.adsorption_constant("inh")

        theta_star = 1.0 + k_acetylene * pa + k_ethylene * pb + k_hydrogen * ph2

        r1_raw = self._k1 * k_acetylene * pa * k_hydrogen * ph2 / theta_star**2
        # Saturating cap. Strictly monotone in r1_raw, smooth at the cap.
        r1 = r1_raw / (1.0 + r1_raw / self._r_cap)

        # Ethylene hydrogenation, with acetylene poisoning the site. The
        # denominator is split as the source writes it: the competitive
        # Langmuir term excludes hydrogen, the inhibition term is separate.
        inhibition = (1.0 + k_inh * pa) ** self.params.n_inh
        r2 = (
            self._k2
            * k_ethylene
            * pb
            * k_hydrogen
            * ph2
            / ((1.0 + k_acetylene * pa + k_ethylene * pb) * inhibition)
        )

        zeros = np.zeros_like(ph2)
        return RateResult(
            gas=_gas_stack(
                -r1 - r2,  # H2
                -r1,  # C2H2
                +r1 - r2,  # C2H4
                +r2,  # C2H6
            ),
            surface=np.stack([zeros, zeros, zeros], axis=-1),
        )

    def coverages(
        self,
        pressures: NDArray[np.float64],
        temperature: float,
    ) -> NDArray[np.float64] | None:
        """Competitive Langmuir isotherm on a single site type.

        Recoverable because the source reports `K` per species instead of only
        as products. The sum is 1, so the vacant-site fraction needs no separate
        balance.
        """
        p = pressures
        ph2 = p[..., IGAS["H2"]]
        pa = p[..., IGAS["C2H2"]]
        pb = p[..., IGAS["C2H4"]]

        w_h2 = self.params.adsorption_constant("H2") * ph2
        w_c2h2 = self.params.adsorption_constant("C2H2") * pa
        w_c2h4 = self.params.adsorption_constant("C2H4") * pb
        denom = 1.0 + w_h2 + w_c2h2 + w_c2h4
        return np.stack([w_h2 / denom, w_c2h2 / denom, w_c2h4 / denom], axis=-1)


# --- Borodzinski & Cybulski 2000 --------------------------------------------


@dataclass(frozen=True, slots=True)
class Borodzinski2000Rate:
    """Two-site model: narrow A sites and wide E sites.

    The mechanism this encodes is geometric, not electronic. Carbonaceous
    deposits on the palladium leave crevices wide enough for acetylene to adsorb
    perpendicularly and for hydrogen atoms to move in, but too narrow for
    ethylene to lie flat. Acetylene is therefore hydrogenated to ethylene on
    sites ethylene cannot occupy, and only the small fraction that does reach an
    E site is over-hydrogenated to ethane.

    That is also why the model has no `A`-site constant for ethylene, and why
    the acetylene-to-ethylene adsorption ratio on E sites is about 2000: trace
    acetylene displaces ethylene and shuts down the undesired reaction. The
    dimensionless ratio `K_C2H2_E / K_C2H4_E` is the single most important
    number in the mechanism, and it is a *ratio*, so unlike the individual
    constants it survives a pressure-unit error.
    """

    params: Borodzinski2000
    site_basis: float = 1.0
    """Metal site density [mol_Pd/kg_cat], injected by the model."""

    @property
    def name(self) -> str:
        return "Borodzinski2000 (two-site LHHW, A/E site populations)"

    def _k(self, index: int) -> float:
        return self.params.rate_constant(f"k{index}")

    def rates(
        self,
        pressures: NDArray[np.float64],
        temperature: float,
    ) -> RateResult:
        p = pressures
        ph2 = p[..., IGAS["H2"]]
        pa = p[..., IGAS["C2H2"]]
        pb = p[..., IGAS["C2H4"]]

        ka = self.params.adsorption_constant("C2H2_A")
        ke_a = self.params.adsorption_constant("C2H2_E")
        ke_b = self.params.adsorption_constant("C2H4_E")

        # A-site isotherm: acetylene only.
        a_denom = 1.0 + ka * pa
        a1 = a_denom**1
        a2 = a_denom**2
        a3 = a_denom**3
        # E-site isotherm: everything competes.
        e_denom = 1.0 + ke_a * pa + ke_b * pb
        e2 = e_denom**2

        # r1: acetylene -> ethane on A sites, three-point adsorption.
        r1 = self._k(1) * pa * ph2**2 / a3
        # r2: ethylene -> ethane on E sites.
        r2 = self._k(2) * pb * ph2 / e2
        # r3: acetylene -> ethylene by non-competitive hydrogen transfer.
        r3 = self._k(3) * pa * ph2 / a1
        # r4: acetylene -> ethylene, competitive, two-point adsorption.
        r4 = self._k(4) * pa * ph2 / a2
        # r5: acetylene dimerisation / oligomerisation to C4.
        r5 = self._k(5) * pa**2 * ph2 / a3

        # Carbon balance. Each C2 carries two carbons, so ethane formation from
        # either C2H2 or C2H4 gives one ethane molecule.
        c2h4_net = r3 + r4 + 2.0 * r5 - r2
        c2h2_net = -(r1 + r3 + r4 + 2.0 * r5)
        c2h6_net = r1 + r2
        h2_net = -(2.0 * r1 + r2 + r3 + r4 + r5)

        zeros = np.zeros_like(ph2)
        return RateResult(
            gas=_gas_stack(h2_net, c2h2_net, c2h4_net, c2h6_net),
            surface=np.stack([zeros, zeros, zeros], axis=-1),
        )

    def coverages(
        self,
        pressures: NDArray[np.float64],
        temperature: float,
    ) -> NDArray[np.float64] | None:
        """Not recoverable, and the reason is worth stating.

        The published constants are products: `k1 = k8 K_C2H2^A K5 K7 (K_H2^A)^2`
        and so on. The elementary constants are not reported, so there is more
        than one set of coverages consistent with the same `k`. Inverting the
        published denominators would silently assume one particular choice of
        elementary constants, which is a claim the source does not make.
        """
        return None


# --- Urmes et al. 2019 ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Urmes2019Rate:
    """Single-site Horiuti-Polanyi sequential hydrogen addition.

    Both transformations run through adsorbed intermediates: acetylene adsorbs,
    takes one hydrogen, takes a second; ethylene adsorbs and does the same.
    Ethylene is the *intermediate* of the wanted reaction and the *reactant* of
    the unwanted one, which is what makes this chemistry hard: the same
    species has to be made fast and consumed slow.

    Dissociative adsorption is visible in the square root on hydrogen in the
    denominator but not in the numerator. That asymmetry is the source's, not a
    typo introduced here -- hydrogen is first adsorbs atomically onto a site
    pair, so coverage goes as `sqrt(p)`, while the reacting step sees a
    hydrogen-covered surface whose dependence enters through the total coverage
    term.

    This is the only set with activation energies, so it is the only one whose
    rates respond to temperature. `arrhenius` is evaluated at the temperature
    passed in, centred on the temperature the set was fitted at.
    """

    params: Urmes2019

    @property
    def name(self) -> str:
        return "Urmes2019 (single-site Horiuti-Polanyi LHHW, Arrhenius)"

    def rates(
        self,
        pressures: NDArray[np.float64],
        temperature: float,
    ) -> RateResult:
        # `arrhenius` returns mol/(mol_Pd_s s); multiplying by the reported
        # site density gives mol/(kg_cat s), the model's working unit.
        k4 = self.params.arrhenius("k04", temperature) * self.params.site_density
        k7 = self.params.arrhenius("k07", temperature) * self.params.site_density

        p = np.maximum(pressures, 0.0)
        ph2 = p[..., IGAS["H2"]]
        pa = p[..., IGAS["C2H2"]]
        pb = p[..., IGAS["C2H4"]]

        sqrt_h2 = np.sqrt(max(self.params.K_H2, 0.0) * ph2)
        k_c2h2 = self.params.K_C2H2
        desorption = pb / self.params.K_C2H4_des

        denom = 1.0 + sqrt_h2 + k_c2h2 * pa + desorption
        denom2 = denom**2

        # Acetylene hydrogenation to ethylene. `k4` arrives already multiplied
        # by N_S, so this is `N_S k04 exp(-E4/RT) K_1 K_2 P P / theta_star^2`.
        r1 = k4 * self.params.K_H2 * k_c2h2 * pa * ph2 / denom2
        # Ethylene hydrogenation to ethane. The source carries an explicit
        # `(K_1 / K_5)` factor, which is *not* folded into `k07`: `k07` is a
        # bare pre-exponential and the source's own expression keeps the ratio
        # separate. Dropping it -- as an earlier version of this file did,
        # having assumed the constant was composite like Borodzinski's -- costs
        # a factor of `K_1 / K_5 = 1.8 Pa^-1` on every ethane rate, which is
        # precisely the reaction whose selectivity is the point of the study.
        r2 = k7 * (self.params.K_H2 / self.params.K_C2H4_des) * pb * ph2 / denom2

        zeros = np.zeros_like(ph2)
        return RateResult(
            gas=_gas_stack(
                -(r1 + r2),
                -r1,
                +(r1 - r2),
                +r2,
            ),
            surface=np.stack([zeros, zeros, zeros], axis=-1),
        )

    def coverages(
        self,
        pressures: NDArray[np.float64],
        temperature: float,
    ) -> NDArray[np.float64] | None:
        """Langmuir coverages, directly from the reported adsorption constants.

        The source publishes the operating-range coverages alongside the rate
        data, so this is directly checkable against them -- one of the few
        places in the model where a computed quantity has an independent
        published value to be wrong against.
        """
        p = pressures
        ph2 = p[..., IGAS["H2"]]
        pa = p[..., IGAS["C2H2"]]
        pb = p[..., IGAS["C2H4"]]

        w_h = np.sqrt(self.params.K_H2 * ph2)
        w_c2h2 = self.params.K_C2H2 * pa
        w_c2h4 = pb / self.params.K_C2H4_des
        denom = 1.0 + w_h + w_c2h2 + w_c2h4
        return np.stack([w_h / denom, w_c2h2 / denom, w_c2h4 / denom], axis=-1)


def rate_law(
    params: KineticSet,
    site_basis: float = 1.0,
    temperature: float | None = None,
) -> RateLaw:
    """Build the rate law matching a parameter set.

    `site_basis` is the metal site density for sets published per mole of metal
    and is ignored by sets published per kilogram of catalyst.
    """
    match params:
        case Larsen2024():
            return Larsen2024Rate(params)
        case Borodzinski2000():
            if site_basis <= 0.0:
                raise ValueError(
                    "Borodzinski2000 rates are per mole of palladium; the "
                    "catalyst declares no metal loading, so they cannot be "
                    "converted to a volumetric rate."
                )
            return Borodzinski2000Rate(params, site_basis=site_basis)
        case Urmes2019():
            return Urmes2019Rate(params)
    raise TypeError(f"no rate law implemented for {type(params).__name__}")


__all__ = [
    "Borodzinski2000Rate",
    "Larsen2024Rate",
    "RateLaw",
    "RateResult",
    "Urmes2019Rate",
    "rate_law",
]
