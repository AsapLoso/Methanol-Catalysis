"""Integration and unit tests for the transient packed-bed reactor solver."""

from __future__ import annotations

import numpy as np

from pbr.model import Inputs, build_model, constant_inlet
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
    """Model must integrate stably forward in time without NaNs or divergence."""
    from pbr.parameters import SolverSettings

    inputs = Inputs.catalogued("urmes2019")
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

    y0 = np.full(model.layout.size, 1e-6)
    span = (0.0, 0.005)
    sol = model.integrate(y0, span, method="Radau")

    assert sol.success, f"Transient integration failed: {sol.message}"
    assert not np.isnan(sol.y).any(), "NaNs detected in solution trajectory"
    assert (sol.y >= -1e-8).all(), (
        f"Negative concentrations detected: min={sol.y.min()}"
    )
