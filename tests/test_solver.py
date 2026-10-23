"""Integration and unit tests for the transient packed-bed reactor solver."""

from __future__ import annotations

import numpy as np

from pbr.chemistry import RateResult
from pbr.model import Inputs, build_model, constant_inlet, make_inlet
from pbr.parameters import R_GAS, fuller_diffusivity
from pbr.species import GASES, IGAS


def test_rhs_does_not_mutate_its_input() -> None:
    """The solver's RHS function must not mutate its input state vector."""
    inputs = Inputs.catalogued("urmes2019")
    # Small test grid for speed
    inputs = (
        inputs._replace(solver=inputs.solver._replace(n_z=4, n_radial=5))
        if hasattr(inputs, "_replace")
        else inputs
    )
    # Rebuild with smaller solver settings
    from pbr.parameters import SolverSettings

    inputs = Inputs(
        geometry=inputs.geometry,
        catalyst=inputs.catalyst,
        operating=inputs.operating,
        kinetics=inputs.kinetics,
        solver=SolverSettings(n_axial=4, n_radial=5),
        radial=inputs.radial,
        external=inputs.external,
        surface=inputs.surface,
    )
    inlet = lambda t: constant_inlet(inputs.operating)
    model = build_model(inputs, inlet)

    y0 = np.linspace(0.1, 1.0, model.layout.size)
    y_copy = y0.copy()

    _ = model.rhs(0.0, y0)
    np.testing.assert_array_equal(
        y0, y_copy, err_msg="RHS mutated input state vector y"
    )


from pbr.species import FORMULAS


def test_atom_conservation_in_chemistry() -> None:
    """Catalytic reaction rates must conserve carbon and hydrogen atoms."""
    inputs = Inputs.catalogued("urmes2019")
    inlet = lambda t: constant_inlet(inputs.operating)
    model = build_model(inputs, inlet)

    c_in = constant_inlet(inputs.operating)
    rt = 8.314462618 * inputs.operating.temperature
    p_gas = c_in * rt

    res = model.rate.rates(p_gas, inputs.operating.temperature)
    gas_rates = res.gas

    c_atoms = sum(gas_rates[IGAS[sp]] * FORMULAS[sp].counts.get("C", 0) for sp in GASES)
    h_atoms = sum(gas_rates[IGAS[sp]] * FORMULAS[sp].counts.get("H", 0) for sp in GASES)

    assert abs(c_atoms) < 1e-12, f"Carbon atom imbalance: {c_atoms}"
    assert abs(h_atoms) < 1e-12, f"Hydrogen atom imbalance: {h_atoms}"


def test_transient_integration_step() -> None:
    """A feed-consistent initial state integrates through a short transient."""
    from pbr.parameters import SolverSettings

    inputs = Inputs.catalogued("urmes2019")
    inputs = Inputs(
        geometry=inputs.geometry,
        catalyst=inputs.catalyst,
        operating=inputs.operating,
        kinetics=inputs.kinetics,
        solver=SolverSettings(
            n_axial=4, n_radial=5, rtol=1e-4, atol=1e-8, advection_scheme="tvd2"
        ),
        radial=inputs.radial,
        external=inputs.external,
        surface=inputs.surface,
    )
    inlet = lambda t: constant_inlet(inputs.operating)
    model = build_model(inputs, inlet)

    c_in = constant_inlet(inputs.operating)
    c_bulk = np.broadcast_to(c_in, (model.layout.n_z, len(GASES))).copy()
    c_particle = np.broadcast_to(
        c_in[None, None, :],
        (model.layout.n_z, model.layout.n_radial, len(GASES)),
    ).copy()
    y0 = model.layout.pack(c_bulk, c_particle)
    sol = model.integrate(y0, (0.0, 0.005), method="Radau", max_step=0.0002)

    assert sol.success, f"Transient integration failed: {sol.message}"
    assert np.isfinite(sol.y).all(), "Non-finite state detected in trajectory"
    assert sol.y.min() >= -1e-7, f"large negative concentration: min={sol.y.min()}"


def test_periodic_tvd2_run_stays_bounded_over_four_cycles() -> None:
    """A smooth forced feed does not trigger a nonconservative TVD overshoot."""
    from pbr.parameters import SolverSettings

    catalogued = Inputs.catalogued("urmes2019")
    frequency = 200.0
    period = 1.0 / frequency
    inputs = Inputs(
        geometry=catalogued.geometry,
        catalyst=catalogued.catalyst,
        operating=catalogued.operating,
        kinetics=catalogued.kinetics,
        solver=SolverSettings(
            n_axial=3,
            n_radial=3,
            rtol=1e-3,
            atol=1e-7,
            advection_scheme="tvd2",
        ),
        radial=catalogued.radial,
        external=catalogued.external,
        surface=catalogued.surface,
    )
    model = build_model(inputs, make_inlet(inputs.operating, [(frequency, 0.01)]))
    c_in = constant_inlet(inputs.operating)
    c_bulk = np.broadcast_to(c_in, (model.layout.n_z, len(GASES))).copy()
    c_particle = np.broadcast_to(
        c_in[None, None, :],
        (model.layout.n_z, model.layout.n_radial, len(GASES)),
    ).copy()
    y0 = model.layout.pack(c_bulk, c_particle)

    sol = model.integrate(
        y0,
        (0.0, 4.0 * period),
        method="Radau",
        max_step=period / 8.0,
    )

    assert sol.success, sol.message
    assert np.isfinite(sol.y).all()
    assert sol.y.min() >= -1e-6
    assert sol.y.max() < 100.0


def test_resolved_pellet_exchange_conserves_bulk_plus_pore_inventory() -> None:
    """No-reaction radial transfer only moves gas between the two phases."""
    from pbr.parameters import SolverSettings

    class ZeroRate:
        name = "zero test rate"

        def rates(self, pressures: np.ndarray, temperature: float) -> RateResult:
            del temperature
            return RateResult(
                gas=np.zeros_like(pressures),
                surface=np.zeros((*pressures.shape[:-1], 3)),
            )

        def coverages(
            self, pressures: np.ndarray, temperature: float
        ) -> np.ndarray | None:
            del pressures, temperature
            return None

    catalogued = Inputs.catalogued("urmes2019")
    inputs = Inputs(
        geometry=catalogued.geometry,
        catalyst=catalogued.catalyst,
        operating=catalogued.operating,
        kinetics=catalogued.kinetics,
        solver=SolverSettings(n_axial=4, n_radial=5),
        radial=catalogued.radial,
        external=catalogued.external,
        surface=catalogued.surface,
    )
    model = build_model(inputs, lambda _time: constant_inlet(inputs.operating))
    model.rate = ZeroRate()

    c_in = constant_inlet(inputs.operating)
    c_bulk = np.broadcast_to(c_in, (model.layout.n_z, len(GASES))).copy()
    radial_shape = np.linspace(0.2, 1.6, model.layout.n_radial)
    c_particle = np.broadcast_to(
        radial_shape[None, :, None] * (c_in[None, None, :] + 0.2),
        (model.layout.n_z, model.layout.n_radial, len(GASES)),
    ).copy()
    derivative = model.rhs(0.0, model.layout.pack(c_bulk, c_particle))
    d_bulk, d_particle, _ = model.layout.unpack(derivative)

    eps_b = model.derived.void_fraction
    eps_p = model.derived.particle_porosity
    total_inventory_derivative = eps_b * d_bulk.sum(axis=0) + (
        (1.0 - eps_b) * eps_p * np.einsum("r,zri->i", model.grid.weights, d_particle)
    )
    np.testing.assert_allclose(total_inventory_derivative, 0.0, atol=1e-10)


def test_finite_radial_reaction_is_only_a_pellet_gas_source() -> None:
    """A uniform bulk/pellet state has no direct duplicate bulk reaction."""
    from pbr.parameters import R_GAS, SolverSettings

    catalogued = Inputs.catalogued("urmes2019")
    inputs = Inputs(
        geometry=catalogued.geometry,
        catalyst=catalogued.catalyst,
        operating=catalogued.operating,
        kinetics=catalogued.kinetics,
        solver=SolverSettings(n_axial=4, n_radial=5),
        radial=catalogued.radial,
        external=catalogued.external,
        surface=catalogued.surface,
    )
    model = build_model(inputs, lambda _time: constant_inlet(inputs.operating))
    c_in = constant_inlet(inputs.operating)
    c_bulk = np.broadcast_to(c_in, (model.layout.n_z, len(GASES))).copy()
    c_particle = np.broadcast_to(
        c_in[None, None, :],
        (model.layout.n_z, model.layout.n_radial, len(GASES)),
    ).copy()

    derivative = model.rhs(0.0, model.layout.pack(c_bulk, c_particle))
    d_bulk, d_particle, _ = model.layout.unpack(derivative)
    local_rate = model.rate.rates(
        c_in * (R_GAS * inputs.operating.temperature), inputs.operating.temperature
    ).gas

    assert np.max(np.abs(local_rate)) > 0.0
    np.testing.assert_allclose(d_bulk, 0.0, atol=1e-10)
    np.testing.assert_allclose(
        d_particle,
        np.broadcast_to(
            model.derived.particle_source_factor * local_rate[None, None, :],
            d_particle.shape,
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_fuller_diffusivity_uses_pa_input_but_atm_correlation_pressure() -> None:
    """The Fuller formula returns gas diffusivities on a realistic SI scale."""
    base = fuller_diffusivity(
        temperature=324.15,
        pressure=126000.0,
        mass_a=2.016,
        mass_b=39.948,
        vol_a=7.07,
        vol_b=16.1,
    )
    double_pressure = fuller_diffusivity(
        temperature=324.15,
        pressure=252000.0,
        mass_a=2.016,
        mass_b=39.948,
        vol_a=7.07,
        vol_b=16.1,
    )

    assert 5.0e-5 < base < 1.0e-4
    assert double_pressure == base / 2.0


def test_inert_remains_in_total_pressure_for_reactive_partial_pressures() -> None:
    """Omitting Ar from state does not renormalize source mole fractions."""
    operating = Inputs.catalogued("urmes2019").operating
    partial_pressures = operating.partial_pressures()

    for species in ("H2", "C2H2", "C2H4"):
        expected = operating.mole_fractions[species] * operating.pressure
        np.testing.assert_allclose(partial_pressures[species], expected)
    assert set(partial_pressures) == {"H2", "C2H2", "C2H4"}
    assert np.isclose(
        sum(operating.concentrations().values()) * R_GAS * operating.temperature,
        operating.reacting_mole_fraction * operating.pressure,
    )
