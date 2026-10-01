"""The literature parameter sets, and the provenance record for each number.

Every value here is transcribed from a primary source and carries a
`Provenance`. Values we chose are marked `ASSUMED` and collected by
`Registry.assumed_parameters()`, which the test suite requires to be
non-empty-safe: an assumed parameter is allowed, but it has to be *named*, and
the sensitivity sweep in `docs/worklog.md` has to cover it.

Three sets, and the reason there are three rather than one is that they
disagree in ways that matter:

* `Urmes2019Set` is the only one with activation energies, so it is the only
  one that can answer a question about temperature. It is also the only one
  whose coverages can be reconstructed, which makes it the only one that can
  drive a relaxing surface. **Default.**
* `Borodzinski2000Set` is a two-site mechanism with the strongest mechanistic
  story and the widest published pressure range. Its constants are per mole of
  palladium, not per kilogram of catalyst, so it needs the metal loading to
  become a volumetric rate.
* `Larsen2024Set` is the only set measured on the reactor this thesis models.
  Its rate cap is **not published** -- see the note on `r_1max`.

Two numbers are flagged as known uncertainties rather than quietly patched,
and one suspected uncertainty was investigated and **cleared**:

`Urmes2019Set.K_H2` -- RESOLVED, no unit error
    A first pass made this `K_H2 = 13 Pa^-1` look three orders of magnitude
    wrong: at the pressure used for the smoke test it dominated the Langmuir
    denominator and drove the acetylene coverage above 0.99 where the paper
    reports 0.62-0.95. Both extraction runs agree on `13` with units of
    `Pa^-1` (Table 5, p. 6), so the suspicion fell on our own feed
    composition. Recomputing at the source's standard condition 1 (1.0%
    C2H2, 4.3% H2, 70% C2H4, 24.7% Ar, 1.26 bar) reproduces all three published
    coverage ranges at once:

        theta_C2H2 = 0.917   (published 0.62-0.95)
        theta_H2   = 0.0018  (published 0.002-0.01)
        theta_C2H4 = 0.081   (published 0.04-0.36)

    The unit was right; the operating point was wrong. `tests/test_chemistry.py`
    pins this as a regression test, because it is the cheapest available check
    that three independent constants and a feed composition are all consistent
    with the paper.

`Larsen2024Set.r_1max`
    The source's rate cap is the parameter that makes its model fit, and the
    value never appears in the extraction. It is present in the model as an
    explicit assumption because the source itself describes the cap as
    empirical rather than mechanistic, so there is no principled value to
    recover -- only a range to sweep.

Two further errors were found by the same exercise and are recorded in
`docs/worklog.md` rather than here, because they are bugs that were fixed, not
open questions: a double pressure-unit conversion on `k`, and a centred-Arrhenius
parameterisation applied to a genuine pre-exponential.
"""

from __future__ import annotations

from pbr.parameters import (
    KINETIC_SETS,
    Borodzinski2000,
    Catalyst,
    Geometry,
    Larsen2024,
    OperatingPoint,
    TransportProperties,
    Urmes2019,
)
from pbr.provenance import Provenance, Registry, Source

B2000 = "borodzinskiKineticModelHydrogenation2000"
U2019 = "urmesKineticStudySelective2019"
L2024 = "larsenSemihydrogenationAcetyleneEthylene2024"
B2006 = "borodzinskiSelectiveHydrogenationEthyne2006"


# --- kinetics ---------------------------------------------------------------


def urmes_2019() -> Urmes2019:
    """Single-site LHHW over 0.05 wt% Pd/alpha-Al2O3, tail-end conditions."""
    return Urmes2019(
        reference=U2019,
        pressure_scale=1.0,
        temperature=324.15,
        k04=1.4e8,
        k07=3.6e9,
        E4=48.5,
        E7=54.8,
        K_H2=13.0,
        K_C2H2=107.0,
        K_C2H4_des=7.4,
        site_density=2.0e-3,
    )


def borodzinski_2000() -> Borodzinski2000:
    """Two-site LHHW over 0.038 wt% Pd/alpha-Al2O3 at 343 K, atmospheric."""
    return Borodzinski2000(
        reference=B2000,
        pressure_scale=1.0e3,
        temperature=343.0,
        k1=12.63,
        k2=0.05208,
        k3=8.352,
        k4=44.38,
        k5=566.5,
        K_C2H2_A=63.92,
        K_C2H2_E=22.23,
        K_C2H4_E=0.01108,
        site_basis=3.5706e-3,
    )


def larsen_2024() -> Larsen2024:
    """Extended LHHW over Pd2Ag/SiO2 at 373.15 K, 1 bar.

    `r_1max` is the one assumed value in this set. Its value is chosen so the
    cap first bites at roughly the acetylene partial pressure at which the
    source's fitted conversion stops rising -- but "roughly" is the honest word,
    and the sweep is what actually licenses the number.
    """
    return Larsen2024(
        reference=L2024,
        pressure_scale=1.0e5,
        temperature=373.15,
        k1=7.1852e-5,
        k2=1.0492e-5,
        r_1max=2.0e-3,
        K_C2H2=936.21,
        K_C2H4=10.21,
        K_H2=3.3319,
        K_inh=972.0,
        n_inh=1.0,
        mass_scale=1.0e3,
    )


# --- geometry and catalyst --------------------------------------------------


def urmes_catalyst() -> Catalyst:
    """Pellet properties for the Pd/alpha-Al2O3 system the diagnostics apply to.

    The particle diameter is the source's own sieve fraction, reduced to a
    radius at the midpoint. Porosity and density are **not** from that source:
    they are carried from a different alumina-based system in the same
    literature, so they are marked assumed rather than cited as that system's.
    """
    return Catalyst(
        name="0.05 wt% Pd / alpha-Al2O3",
        support="alpha-Al2O3",
        particle_radius=75.0e-6,
        particle_porosity=0.50,
        particle_density=1550.0,
        tortuosity=3.0,
        metal_mass_fraction=5.0e-4,
        site_density=2.0e-3,
    )


def larsen_catalyst() -> Catalyst:
    """Pellet properties for the PdAg/silica bed of the reference reactor."""
    return Catalyst(
        name="0.05 wt% Pd / 5 wt% Ag on SiO2",
        support="SiO2",
        particle_radius=75.0e-6,
        particle_porosity=0.45,
        particle_density=2650.0,
        tortuosity=3.0,
        metal_mass_fraction=5.0e-4,
    )


def urmes_geometry() -> Geometry:
    """Tube and bed sized to the reference reactor's footprint.

    Diameter is the source's; the catalyst mass is ours, chosen to give a bed
    with the same order of length at the reference space velocity.
    """
    return Geometry(
        tube_inner_diameter=5.0e-3,
        catalyst_mass=7.5e-5,
        bed_void_fraction=0.45,
    )


def urmes_operating() -> OperatingPoint:
    """Feed at the source's standard condition 1 (Table 1, p. 4).

    The composition is quoted as a *volume fraction of the whole stream*,
    which is how the source prints it: 1.0% C2H2, 4.3% H2, 70% C2H4, 24.7% Ar.
    Ar is carried as the inert.

    Getting this composition right is not cosmetic. The reported coverages
    (theta_C2H2 = 0.62-0.95, theta_H2 = 0.002-0.01, theta_C2H4 = 0.04-0.36)
    are the only independent check on the Langmuir terms this model can make,
    and they only come out if the partial pressures are the source's.
    """
    return OperatingPoint(
        temperature=324.15,
        pressure=126000.0,
        volumetric_flow=200.0e-6 / 60.0,
        mole_fractions={"C2H2": 0.010, "C2H4": 0.70, "H2": 0.043, "Ar": 0.247},
    )


def borodzinski_operating() -> OperatingPoint:
    """Feed chosen to land inside the source's measured partial-pressure window.

    The source works at 343 K and atmospheric pressure with
    `p_C2H2 = 0.0035-0.078 kPa`, `p_C2H4 = 0.047-40 kPa`,
    `p_H2 = 0.55-0.78 kPa`. Those partial pressures are the model's real
    domain; a convenient-looking round feed would sit outside it, and the rate
    laws extrapolate badly well outside a fitted window -- an LHHW denominator
    extrapolated from far above its coverage range is not a measurement, it is
    a guess with a decimal point in it.

    Composition is therefore derived from the target partial pressures at
    101325 Pa rather than chosen for neatness.
    """
    total = 101325.0
    target = {"C2H2": 20.0, "C2H4": 20_000.0, "H2": 640.0}
    fractions = {name: p / total for name, p in target.items()}
    fractions["Ar"] = 1.0 - sum(fractions.values())
    return OperatingPoint(
        temperature=343.0,
        pressure=total,
        volumetric_flow=20.0e-6 / 60.0,
        mole_fractions=fractions,
    )


def larsen_operating() -> OperatingPoint:
    """Feed at the source's baseline: 1/20/20 in helium."""
    return OperatingPoint(
        temperature=373.15,
        pressure=101325.0,
        volumetric_flow=100.0e-6 / 60.0,
        mole_fractions={"C2H2": 0.01, "C2H4": 0.20, "H2": 0.20, "He": 0.59},
    )


# --- provenance -------------------------------------------------------------

PROVENANCE = Registry(
    {
        # -- Urmes 2019 kinetics
        "urmes.k04": Provenance(U2019, "Table 3, k04", note="mol/(mol Pd_s s)"),
        "urmes.k07": Provenance(U2019, "Table 3, k07", note="mol/(mol Pd_s s)"),
        "urmes.E4": Provenance(U2019, "Table 3, E4", note="kJ/mol"),
        "urmes.E7": Provenance(U2019, "Table 3, E7", note="kJ/mol"),
        "urmes.K_H2": Provenance(
            U2019,
            "Table 5, p. 6",
            note=(
                "Pa^-1, verified: reproduces all three published coverage ranges "
                "at the standard operating condition"
            ),
        ),
        "urmes.K_C2H2": Provenance(U2019, "Table 5, p. 6", note="107 +/- 7 Pa^-1"),
        "urmes.K_C2H4_des": Provenance(U2019, "Table 5, p. 6", note="7.4 Pa (fixed)"),
        "urmes.site_density": Provenance(U2019, "Table 1, N_S", note="mol Pd_s/kg"),
        # -- Borodzinski 2000 kinetics
        "borodzinski.k1": Provenance(B2000, "Table 3", note="mol/mol_Pd/s/kPa^-3"),
        "borodzinski.k2": Provenance(B2000, "Table 3", note="mol/mol_Pd/s/kPa^-2"),
        "borodzinski.k3": Provenance(B2000, "Table 3", note="mol/mol_Pd/s/kPa^-2"),
        "borodzinski.k4": Provenance(B2000, "Table 3", note="mol/mol_Pd/s/kPa^-2"),
        "borodzinski.k5": Provenance(B2000, "Table 3", note="mol/mol_Pd/s/kPa^-3"),
        "borodzinski.K_C2H2_A": Provenance(B2000, "Table 3", note="kPa^-1"),
        "borodzinski.K_C2H2_E": Provenance(B2000, "Table 3", note="kPa^-1"),
        "borodzinski.K_C2H4_E": Provenance(B2000, "Table 3", note="kPa^-1"),
        "borodzinski.site_basis": Provenance(
            B2000,
            "Table 1, 0.038 wt% Pd",
            Source.DERIVED,
            note="0.00038 / 106.42e-3 kg/mol = 3.57e-3 mol Pd/kg_cat",
        ),
        # -- Larsen 2024
        "larsen.k1": Provenance(L2024, "Appendix F, fitted set", note="mol/g_cat/s"),
        "larsen.k2": Provenance(L2024, "Appendix F, fitted set", note="mol/g_cat/s"),
        "larsen.r_1max": Provenance(
            L2024,
            "not published",
            Source.ASSUMED,
            note="the cap's value is absent from the source; swept in docs/worklog.md",
        ),
        "larsen.K_C2H2": Provenance(L2024, "Appendix F", note="bar^-1"),
        "larsen.K_C2H4": Provenance(L2024, "Appendix F", note="bar^-1"),
        "larsen.K_H2": Provenance(L2024, "Appendix F", note="bar^-1"),
        "larsen.K_inh": Provenance(L2024, "Appendix F", note="bar^-1"),
        "larsen.n_inh": Provenance(L2024, "Appendix F", note="exponent, 1.0 = linear"),
        # -- geometry
        "catalyst.particle_radius": Provenance(
            U2019,
            "Table 1, sieve fraction 100-200 um",
            Source.DERIVED,
            note="radius = 150 um / 2 = 75 um, the midpoint of the fraction",
        ),
        "catalyst.particle_porosity": Provenance(
            "bergerDynamicMethodsCatalytic2008",
            "Case 6.4 pellet porosity",
            Source.ASSUMED,
            note="transferred from an Na/gamma-Al2O3 sorbent, not from this catalyst",
        ),
        "catalyst.particle_density": Provenance(
            "bergerDynamicMethodsCatalytic2008",
            "Case 6.4 particle density",
            Source.ASSUMED,
            note="1550 kg/m3, an alumina pellet from a different study",
        ),
        "catalyst.tortuosity": Provenance(
            "dehghaniExperimentalApproachIndustrial2019",
            "reactor description",
            Source.ASSUMED,
            note="tau = 2.5 reported there; 3.0 used here as an alumina-typical value",
        ),
        "catalyst.metal_mass_fraction": Provenance(
            U2019, "Table 1", note="0.05 wt% Pd"
        ),
        "geometry.tube_inner_diameter": Provenance(L2024, "Appendix F", note="5.0 mm"),
        "geometry.catalyst_mass": Provenance(
            L2024,
            "Appendix F, W_cat",
            Source.ASSUMED,
            note="75 mg is the source's Pd2Ag bed; kept for the reactor match",
        ),
        "geometry.bed_void_fraction": Provenance(
            "bergerDynamicMethodsCatalytic2008",
            "Case 6.4",
            Source.ASSUMED,
            note="0.50 reported for a packed alumina bed of a different study",
        ),
        # -- operating point
        "operating.temperature": Provenance(
            U2019, "Section 2", note="51 degC = 324.15 K"
        ),
        "operating.pressure": Provenance(U2019, "Section 2", note="1.26 bar"),
        "operating.volumetric_flow": Provenance(U2019, "Section 2", note="200 mL/min"),
        "operating.mole_fractions": Provenance(
            U2019, "Section 2", note="feed composition"
        ),
        # -- transport
        "transport.effective_diffusivity": Provenance(
            "fullerSchettlerGiddings1966",
            "Eq. for gas-phase diffusivity",
            Source.DERIVED,
            note=(
                "D_e = eps_p D_m / tau with D_m from the Fuller correlation; "
                "this reference is not yet in references.bib"
            ),
        ),
        "transport.external_film": Provenance(
            "urmesKineticStudySelective2019",
            "Table 7, Carberry 3.3e-4",
            Source.DERIVED,
            note="Ca << 0.05 at this catalyst, so the film closure is inactive by default",
        ),
        "transport.axial_dispersion": Provenance(
            "danckwertsContinuousFlowSystems1953",
            "Eq. 30",
            Source.DERIVED,
            note="D_Lu = 0.018 of u_s*d_p; the closure defaults to zero (plug flow)",
        ),
        # -- surface dynamics
        "surface.tau": Provenance(
            "not in the parameter database",
            "n/a",
            Source.ASSUMED,
            note=(
                "no source gives a surface relaxation time; swept over orders of "
                "magnitude in docs/worklog.md"
            ),
        ),
    }
)


#: Sensitivity sweep for every parameter the model had to choose.
#: `docs/worklog.md` reports the outcome; nothing may be concluded from a run
#: that has not covered these.
ASSUMED_SWEEP = {
    "larsen.r_1max": [2.0e-4, 2.0e-3, 2.0e-2],
    "catalyst.particle_porosity": [0.40, 0.50, 0.60],
    "catalyst.particle_density": [1550.0, 2000.0, 2650.0],
    "catalyst.tortuosity": [2.0, 3.0, 5.0],
    "geometry.bed_void_fraction": [0.40, 0.45, 0.50],
    "surface.tau": [1.0e-3, 1.0e-2, 1.0e-1, 1.0],
}

__all__ = [
    "ASSUMED_SWEEP",
    "KINETIC_SETS",
    "PROVENANCE",
    "TransportProperties",
    "borodzinski_2000",
    "larsen_2024",
    "larsen_catalyst",
    "larsen_operating",
    "urmes_2019",
    "urmes_catalyst",
    "urmes_geometry",
    "urmes_operating",
]
