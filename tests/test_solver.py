"""Integration and unit tests for the transient packed-bed reactor solver."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pbr.chemistry import RateResult
from pbr.model import (
    Inputs,
    build_model,
    constant_inlet,
    effective_diffusivities,
    make_inlet,
)
from pbr.parameters import R_GAS, fuller_diffusivity
from pbr.species import FORMULAS, GASES, IGAS


def _small_inputs(n_axial: int = 2, n_radial: int = 3) -> Inputs:
    """Use catalogued science inputs with a test-sized spatial grid."""
    catalogued = Inputs.catalogued("urmes2019")
    solver = catalogued.solver.model_copy(
        update={"n_axial": n_axial, "n_radial": n_radial}
    )
    return replace(catalogued, solver=solver)


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


def test_no_film_surface_is_the_bulk_value_at_the_physical_radial_face() -> None:
    """The no-film boundary is algebraic at R; no surface state is evolved."""
    inputs = _small_inputs(n_axial=3, n_radial=5)
    model = build_model(
        inputs, lambda _time: constant_inlet(inputs.operating)
    )
    c_bulk = np.arange(1, 1 + 3 * len(GASES), dtype=float).reshape(3, len(GASES))
    c_particle = np.full((3, 5, len(GASES)), 2.0)

    surface = model.shell_concentration(c_bulk, c_particle)

    np.testing.assert_array_equal(surface, c_bulk)
    assert model.grid.r[-1] < model.grid.r_p
    assert not model.layout.carries_surface


def test_reactive_bulk_pore_inventory_has_exactly_one_pellet_source() -> None:
    """Summed inventories equal the radial-volume integral of one reaction."""
    inputs = _small_inputs(n_axial=3, n_radial=5)
    model = build_model(
        inputs, lambda _time: constant_inlet(inputs.operating)
    )
    c_feed = constant_inlet(inputs.operating)
    c_bulk = np.broadcast_to(c_feed, (3, len(GASES))).copy()
    radial_scale = np.linspace(0.8, 1.2, model.layout.n_radial)[None, :, None]
    c_particle = np.broadcast_to(
        (c_feed[None, None, :] + 0.3) * radial_scale,
        (3, model.layout.n_radial, len(GASES)),
    ).copy()

    derivative = model.rhs(0.0, model.layout.pack(c_bulk, c_particle))
    d_bulk, d_particle, _ = model.layout.unpack(derivative)
    rates = model.rate.rates(
        c_particle * (R_GAS * inputs.operating.temperature),
        inputs.operating.temperature,
    ).gas
    eps_b = model.derived.void_fraction
    eps_p = model.derived.particle_porosity
    inventory_rate = eps_b * d_bulk.sum(axis=0) + (1.0 - eps_b) * eps_p * np.einsum(
        "r,zri->i", model.grid.weights, d_particle
    )
    single_reaction_source = (1.0 - eps_b) * inputs.catalyst.particle_density * np.einsum(
        "r,zri->i", model.grid.weights, rates
    )

    assert np.max(np.abs(single_reaction_source)) > 0.0
    np.testing.assert_allclose(
        inventory_rate, single_reaction_source, rtol=2e-11, atol=1e-9
    )
    for element in ("C", "H"):
        atom_counts = np.array(
            [FORMULAS[gas].counts.get(element, 0) for gas in GASES], dtype=float
        )
        np.testing.assert_allclose(atom_counts @ inventory_rate, 0.0, atol=1e-9)


def test_sparse_jacobian_mask_covers_finite_difference_nonzeros() -> None:
    """Finite-difference Jacobian structure is included in the sparse mask."""
    inputs = _small_inputs(n_axial=3, n_radial=3)
    model = build_model(
        inputs, lambda _time: constant_inlet(inputs.operating)
    )
    c_feed = constant_inlet(inputs.operating)
    z_scale = (1.0 + 0.04 * np.arange(model.layout.n_z))[:, None]
    c_bulk = (c_feed[None, :] + 0.3) * z_scale
    radial_scale = np.linspace(0.8, 1.2, model.layout.n_radial)[None, :, None]
    c_particle = np.broadcast_to(
        (c_bulk[:, None, :] + 0.2) * radial_scale,
        (model.layout.n_z, model.layout.n_radial, len(GASES)),
    ).copy()
    state = model.layout.pack(c_bulk, c_particle)
    sparsity = model.sparsity.toarray().astype(bool)
    missing: list[tuple[int, int, float]] = []

    for column in range(state.size):
        step = np.cbrt(np.finfo(float).eps) * max(abs(state[column]), 1.0)
        plus, minus = state.copy(), state.copy()
        plus[column] += step
        minus[column] -= step
        derivative = (model.rhs(0.0, plus) - model.rhs(0.0, minus)) / (2.0 * step)
        threshold = max(np.max(np.abs(derivative)) * 1e-8, 1e-10)
        rows = np.flatnonzero(np.abs(derivative) > threshold)
        missing.extend(
            (int(row), column, float(derivative[row]))
            for row in rows
            if not sparsity[row, column]
        )

    assert not missing, f"finite-difference nonzeros omitted by mask: {missing[:10]}"


def test_sparse_and_dense_bdf_agree_on_a_stiff_reactive_transient() -> None:
    """Sparse and dense finite-difference Jacobians give the same trajectory."""
    base = Inputs.catalogued("urmes2019")
    solver = base.solver.model_copy(
        update={
            "n_axial": 2,
            "n_radial": 4,
            "rtol": 1e-8,
            "atol": 1e-12,
            "max_step": 5e-4,
            "integrator": "BDF",
        }
    )
    inputs = replace(base, solver=solver)
    model = build_model(
        inputs, make_inlet(inputs.operating, forcing=[(5.0, 0.10)])
    )
    c_feed = constant_inlet(inputs.operating)
    c_bulk = np.broadcast_to(c_feed, (2, len(GASES))).copy()
    c_particle = np.broadcast_to(
        c_feed[None, None, :], (2, 4, len(GASES))
    ).copy()
    state = model.layout.pack(c_bulk, c_particle)
    common = {
        "y0": state,
        "t_span": (0.0, 0.01),
        "method": "BDF",
        "rtol": 1e-8,
        "atol": 1e-12,
        "max_step": 5e-4,
        "dense_output": True,
    }
    sparse = model.integrate(**common, use_sparsity=True)
    dense = model.integrate(**common, use_sparsity=False)

    assert sparse.success, sparse.message
    assert dense.success, dense.message
    assert sparse.sol is not None and dense.sol is not None
    times = np.linspace(0.0, 0.01, 31)
    np.testing.assert_allclose(
        sparse.sol(times), dense.sol(times), rtol=2e-6, atol=2e-9
    )


def test_all_species_constant_r2_over_d_preserves_integrated_trajectories() -> None:
    """Three R/D pairs preserve RHS and the one-period PSSA trajectory."""
    base = Inputs.catalogued("urmes2019")
    solver = base.solver.model_copy(
        update={
            "n_axial": 2,
            "n_radial": 7,
            "rtol": 1e-7,
            "atol": 1e-12,
            "max_step": 2.5e-3,
            "integrator": "BDF",
        }
    )
    base = replace(base, solver=solver)
    reference_radius = base.catalyst.particle_radius
    reference_diffusivities = effective_diffusivities(base.catalyst, base.operating)
    inlet = make_inlet(base.operating, forcing=[(5.0, 0.10)])
    c_feed = constant_inlet(base.operating)
    c_bulk = np.broadcast_to(c_feed, (2, len(GASES))).copy()
    c_particle = np.broadcast_to(
        c_feed[None, None, :], (2, 7, len(GASES))
    ).copy()
    state = None
    reference_rhs = None
    reference_trajectory = None
    sample_times = np.linspace(0.0, 0.2, 101)

    for radius_factor, diffusivity_factor in ((0.5, 0.25), (1.0, 1.0), (2.0, 4.0)):
        catalyst = base.catalyst.model_copy(
            update={"particle_radius": reference_radius * radius_factor}
        )
        inputs = replace(base, catalyst=catalyst)
        diffusivities = {
            gas: value * diffusivity_factor
            for gas, value in reference_diffusivities.items()
        }
        model = build_model(inputs, inlet, diffusivities=diffusivities)
        if state is None:
            state = model.layout.pack(c_bulk, c_particle)
        rhs = model.rhs(0.0, state)
        if reference_rhs is None:
            reference_rhs = rhs
        else:
            np.testing.assert_allclose(rhs, reference_rhs, rtol=2e-12, atol=1e-10)

        result = model.integrate(
            state,
            (0.0, 0.2),
            method="BDF",
            rtol=1e-7,
            atol=1e-12,
            max_step=2.5e-3,
            use_sparsity=True,
            dense_output=True,
        )
        assert result.success, result.message
        assert result.sol is not None
        trajectory = result.sol(sample_times)
        if reference_trajectory is None:
            reference_trajectory = trajectory
        else:
            np.testing.assert_allclose(
                trajectory, reference_trajectory, rtol=2e-7, atol=2e-9
            )


def test_catalogued_fuller_diffusivities_match_pressure_converted_reference() -> None:
    """The catalogued effective D values use Pa input and atm Fuller pressure."""
    inputs = Inputs.catalogued("urmes2019")
    diffusivities = effective_diffusivities(inputs.catalyst, inputs.operating)
    expected = np.array(
        [
            1.2264424072028763e-5,
            2.44383454036021e-6,
            2.303977722077158e-6,
            2.06131251611772e-6,
        ]
    )
    actual = np.array([diffusivities[gas] for gas in GASES])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)
