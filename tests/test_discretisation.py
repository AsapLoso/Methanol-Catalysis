"""Verification of the discretisation against closed-form solutions.

Everything here checks against something that can be worked out on paper. That
matters more than usual in this model: the parameter database contains no
analytic benchmark for a pellet solver (the audit for the earlier scaffold
recorded "no eta(phi) table, no c(r) profile at known phi, none"), so the only
way to know whether the radial operator is right is to test it against the
textbook solution for the same problem.

The reference solutions are reaction-engineering standard and are derived in
the docstrings where they are used. They are not citations to the parameter
database, and they do not need to be: they are mathematics, and a maths result
is checkable by differentiating it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pbr.closures import (
    AlgebraicEquilibrium,
    DanckwertsDispersion,
    ExplicitDiffusion,
    FilmResistance,
    NoDispersion,
    NoExternalResistance,
    Relaxation,
    Uniform,
    dispersion_for,
    external_specific_area,
)
from pbr.discretisation import (
    FiniteVolumeRadialGrid,
    RadialGrid,
    axial_advection,
    axial_dispersion,
    centre_operator,
    jacobian_sparsity,
    radial_diffusion,
    radial_diffusion_fv,
)

# --- radial diffusion ------------------------------------------------------


@pytest.mark.parametrize("geometry", [0, 1, 2])
def test_finite_volume_radial_operator_closes_on_its_surface_flux(
    geometry: int,
) -> None:
    """The volume-integrated diffusion is exactly the flux at the R face."""
    grid = FiniteVolumeRadialGrid(r_p=2.0, n_nodes=7, geometry=geometry)
    diffusivity, porosity = 3.0e-5, 0.4
    particle = np.stack(
        [np.linspace(0.3, 1.1, grid.n_nodes), np.linspace(2.0, 0.8, grid.n_nodes)]
    )
    shell = np.array([1.4, 0.2])
    derivative = radial_diffusion_fv(particle, grid, diffusivity, porosity, shell)

    average_derivative = derivative @ grid.weights
    expected = (
        diffusivity
        / porosity
        * (geometry + 1)
        / grid.r_p
        * (shell - particle[:, -1])
        / (grid.dr / 2.0)
    )
    np.testing.assert_allclose(average_derivative, expected, rtol=2e-14, atol=1e-14)


def test_finite_volume_radial_operator_keeps_a_uniform_profile_constant() -> None:
    grid = FiniteVolumeRadialGrid(r_p=75.0e-6, n_nodes=9, geometry=2)
    concentration = np.full((4, grid.n_nodes), 1.7)
    shell = np.full(4, 1.7)
    derivative = radial_diffusion_fv(concentration, grid, 1e-6, 0.5, shell)
    np.testing.assert_allclose(derivative, 0.0, atol=1e-12)


def test_spherical_center_face_is_zero_flux_for_a_nonuniform_profile() -> None:
    """With zero surface gradient, integrated radial diffusion is zero."""
    grid = FiniteVolumeRadialGrid(r_p=75.0e-6, n_nodes=7, geometry=2)
    concentration = np.array([[0.8, 1.1, 1.4, 1.2, 0.9, 0.7, 0.6]])
    same_as_outer_cell = concentration[:, -1]

    derivative = radial_diffusion_fv(
        concentration, grid, d_eff=2.0e-6, porosity=0.5,
        shell=same_as_outer_cell,
    )

    np.testing.assert_allclose(derivative @ grid.weights, 0.0, atol=1e-11)


def test_finite_volume_sphere_converges_to_the_first_order_reaction_profile() -> None:
    """The cell-centred grid reproduces the analytic steady pellet profile."""
    errors = []
    phi = 3.0
    for n_cells in (10, 20, 40):
        grid = FiniteVolumeRadialGrid(r_p=1.0, n_nodes=n_cells, geometry=2)
        identity = np.eye(n_cells)
        zero_shell = np.array([0.0])
        operator = np.column_stack(
            [
                radial_diffusion_fv(
                    identity[row : row + 1], grid, 1.0, 1.0, zero_shell
                )[0]
                for row in range(n_cells)
            ]
        )
        boundary = radial_diffusion_fv(
            np.zeros((1, n_cells)), grid, 1.0, 1.0, np.array([1.0])
        )[0]
        profile = np.linalg.solve(operator - phi**2 * identity, -boundary)
        exact = spherical_steady_profile(phi, grid.r / grid.r_p)
        errors.append(float(np.max(np.abs(profile - exact))))

    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert max(errors) < 6e-3
    assert min(ratios) > 3.0, f"finite-volume convergence ratios {ratios}, expected ~4"


def operator_matrix(grid: RadialGrid, d_eff: float, porosity: float) -> np.ndarray:
    """Dense matrix of `radial_diffusion`, by applying it to basis vectors."""
    n = grid.n_nodes
    columns = []
    for j in range(n):
        basis = np.zeros((1, n))
        basis[0, j] = 1.0
        columns.append(radial_diffusion(basis, grid, d_eff, porosity)[0])
    return np.column_stack(columns)


def spherical_steady_profile(phi: float, r_over_rp: np.ndarray) -> np.ndarray:
    """Closed-form `c/c_s` for a first-order reaction in a sphere.

    Steady state is `0 = D_eff (1/r^2) d/dr (r^2 dc/dr) - k_v c`. Substituting
    `c = A sinh(phi r/R)/r` solves it; the regular solution at the origin is
    `sinh(phi r/R)/r`, and normalising so that `c(R) = c_s` gives

        c(r)/c_s = (R / (r sinh phi)) sinh(phi r / R)

    which is what this returns, with the removable singularity at `r = 0`
    filled by its limit `phi / sinh(phi)`.
    """
    x = np.asarray(r_over_rp, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        value = 1.0 / (x * np.sinh(phi)) * np.sinh(phi * x)
    return np.where(np.abs(x) < 1e-14, phi / np.sinh(phi), value)


def effectiveness(phi: float) -> float:
    """Closed-form effectiveness factor for first order in a sphere.

    The ratio of the actual volumetric rate to the rate that would occur if the
    whole pellet were at the surface concentration, obtained by integrating the
    profile above over the volume:

        eta = 3 / phi^2 (phi coth phi - 1)
    """
    if phi < 1e-8:
        return 1.0
    return float(3.0 / phi**2 * (phi / np.tanh(phi) - 1.0))


def solve_first_order_sphere(
    n_radial: int, phi: float, geometry: int = 2
) -> tuple[np.ndarray, RadialGrid]:
    """Steady `c/c_s` profile on the discrete grid, by a linear solve."""
    grid = RadialGrid(r_p=1.0, n_nodes=n_radial, geometry=geometry)
    d_eff, porosity = 1.0, 1.0
    operator = operator_matrix(grid, d_eff, porosity)
    # phi^2 = k_v R^2 / D_eff  ->  k_v in the units of the operator.
    sink = phi**2 / grid.r_p**2

    size = grid.n_nodes
    matrix = operator.copy()
    matrix[np.arange(size), np.arange(size)] -= sink
    # The outermost node is a Dirichlet node: the shell value is prescribed,
    # not solved for. Drop its row and move the known term to the right-hand
    # side. The sign matters -- the known c_s appears with a plus in the
    # equation, so it moves across as a minus.
    matrix = matrix[: size - 1, : size - 1]
    rhs = -operator[: size - 1, size - 1]

    interior = np.linalg.solve(matrix, rhs)
    profile = np.concatenate([interior, [1.0]])
    return profile, grid


@pytest.mark.parametrize("n_radial", [10, 20, 40, 80, 160])
def test_first_order_sphere_matches_closed_form(n_radial: int) -> None:
    """The radial operator reproduces the textbook sphere profile.

    The reference is the analytic solution of the same PDE, so this is a real
    check on the flux-form stencil and on the centre-node treatment rather
    than a regression against previously recorded output.
    """
    profile, grid = solve_first_order_sphere(n_radial, phi=3.0)
    exact = spherical_steady_profile(3.0, grid.r / grid.r_p)
    error = np.max(np.abs(profile - exact))
    tol = 1e-2 if n_radial <= 10 else 5e-3
    assert error < tol, f"max error {error:.2e} at N_r={n_radial}"


def test_sphere_profile_converges_at_second_order() -> None:
    """Error must fall like `dr^2`, which is what a central scheme should do.

    Only meaningful if the scheme is genuinely second order. A scheme with the
    wrong centre-node coefficient -- the defect this model was rebuilt to fix --
    converges at first order and would fail this test while still producing a
    curve that looks physically reasonable.
    """
    errors = []
    for n_radial in (10, 20, 40, 80):
        profile, grid = solve_first_order_sphere(n_radial, phi=3.0)
        exact = spherical_steady_profile(3.0, grid.r / grid.r_p)
        errors.append(np.max(np.abs(profile - exact)))
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    # A factor approaching 4 is second order. Allow some slack for the
    # asymptotic regime not being fully reached at N_r = 10.
    assert min(ratios) > 3.0, f"convergence ratios {ratios}, expected ~4"


def test_slab_profile_matches_closed_form() -> None:
    """`c/c_s = cosh(phi x)/cosh(phi)` for a plane slab.

    Kept as a second geometry for the same reason `s` is in the operator: a
    hard-coded 6 would pass every sphere test and be wrong here.
    """
    from scipy.special import iv

    n = 60
    grid = RadialGrid(r_p=1.0, n_nodes=n, geometry=1)
    operator = operator_matrix(grid, 1.0, 1.0)
    sink = 4.0
    matrix = operator.copy()
    matrix[np.arange(n), np.arange(n)] -= sink
    rhs = -operator[: n - 1, n - 1]
    matrix = matrix[: n - 1, : n - 1]
    profile = np.concatenate([np.linalg.solve(matrix, rhs), [1.0]])

    phi = 2.0
    x = grid.r / grid.r_p
    exact = iv(0, phi * x) / iv(0, phi)
    assert np.max(np.abs(profile - exact)) < 5e-3


# --- quadrature ------------------------------------------------------------


@pytest.mark.parametrize("n_nodes", [3, 5, 9, 17, 65])
def test_volume_weights_are_exact_for_the_volume_element(n_nodes: int) -> None:
    """The weights must reproduce volume averages of cubics exactly on any odd grid."""
    for s in (0, 1, 2):
        grid = RadialGrid(r_p=1.0, n_nodes=n_nodes, geometry=s)
        assert float(grid.weights.sum()) == pytest.approx(1.0, abs=1e-14)
        exact_ratio = (s + 1) / 4.0
        discrete = float(np.sum(grid.weights * grid.r ** (3 - s)))
        assert discrete == pytest.approx(exact_ratio, rel=1e-12), (
            f"s={s}, N_r={n_nodes}: weights do not integrate the volume element"
        )


def test_volume_weights_sum_to_one_for_even_node_counts_too() -> None:
    """The normalisation guarantee must not depend on the parity branch."""
    for n_nodes in (4, 10, 24):
        for s in (0, 1, 2):
            grid = RadialGrid(r_p=1.0, n_nodes=n_nodes, geometry=s)
            assert float(grid.weights.sum()) == pytest.approx(1.0, abs=1e-14)


def test_even_node_counts_converge_quadratically() -> None:
    """Trapezoid fallback must still improve, just not as fast as Simpson."""
    errors = []
    for n in (16, 32, 64, 128):
        grid = RadialGrid(r_p=1.0, n_nodes=n, geometry=2)
        analytic = 3.0 / 5.0
        errors.append(abs(float(np.sum(grid.weights * grid.r**2)) - analytic))
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert min(ratios) > 3.5, f"trapezoid fallback ratios {ratios}, expected ~4"


def test_centre_operator_coefficient_is_2_times_1_plus_s() -> None:
    """The exact centre coefficient, isolated from the grid.

    This is the specific regression the rebuild exists to prevent: a sphere
    needs `6`, and the common error is to compute `(1+s) = 3`.
    """
    for s in (0, 1, 2):
        dr = 1e-3
        assert centre_operator(s, dr, 1.0) == pytest.approx(2.0 * (1.0 + s) / dr**2)
    assert centre_operator(2, 1.0, 1.0) == pytest.approx(6.0)
    assert centre_operator(2, 1.0, 1.0) != pytest.approx(3.0)


# --- axial operators -------------------------------------------------------


def test_upwind_advection_is_exact_for_a_constant() -> None:
    """A uniform profile must have zero derivative under any scheme."""
    n_z, n_gas = 10, 1
    dz = 1.0 / n_z
    c = np.full((n_z, n_gas), 3.0)
    out = axial_advection(c, u_s=0.2, dz=dz, c_inlet=np.array([3.0]))
    assert np.max(np.abs(out)) < 1e-14


def test_upwind_advection_sign_and_magnitude() -> None:
    """A decreasing cell-centred profile accumulates at the inlet and downstream."""
    n_z = 6
    dz = 1.0 / n_z
    z = (np.arange(n_z) + 0.5) * dz
    slope = -2.0
    c = (3.0 + slope * z)[:, None]
    out = axial_advection(c, u_s=0.5, dz=dz, c_inlet=np.array([3.0]))[:, 0]
    # dC/dt = -u * dC/dz, so a negative slope gives a positive accumulation.
    assert np.all(out >= 0.0)
    assert out[0] == pytest.approx(0.5, rel=1e-12)  # first-order boundary half-cell
    assert out[-1] == pytest.approx(-0.5 * slope, rel=1e-12)


@pytest.mark.parametrize("scheme", ["upwind1", "tvd2"])
def test_both_schemes_reproduce_a_linear_profile(scheme: str) -> None:
    """First order must be exact for a straight line; the limiter must not clip it.

    A TVD limiter that fires on a smooth ramp is a bug: it silently drops to
    first order everywhere, which is exactly what it is supposed to avoid.
    """
    n_z = 20
    dz = 1.0 / n_z
    z = (np.arange(n_z) + 0.5) * dz
    c = (1.0 - 0.5 * z)[:, None]
    out = axial_advection(c, u_s=0.4, dz=dz, c_inlet=np.array([1.0]), scheme=scheme)
    assert np.max(np.abs(out[1:-1] - 0.4 * 0.5)) < 1e-12


def test_tvd2_is_second_order_and_upwind_is_not() -> None:
    """Demonstrate the order difference the default scheme buys.

    Both are monotone; only one is second order. The test asserts the ordering,
    not an absolute rate, because the point is the *ratio* of their errors.
    """

    def error(scheme: str, n: int) -> float:
        dz = 1.0 / n
        z = (np.arange(n) + 0.5) * dz
        slope = -0.5
        c = 1.0 + slope * z
        bump = 0.05 * np.exp(-((z - 0.5) ** 2) / (2 * 0.05**2))
        perturbed = (c + bump)[:, None]
        bump_inlet = 0.05 * np.exp(-((0.0 - 0.5) ** 2) / (2 * 0.05**2))
        c_inlet = np.array([1.0 + bump_inlet])
        out = axial_advection(perturbed, 0.4, dz, c_inlet, scheme=scheme)
        dbump = (
            -0.05 * (z - 0.5) / (0.05**2) * np.exp(-((z - 0.5) ** 2) / (2 * 0.05**2))
        )
        exact = -0.4 * (slope + dbump)[:, None]
        return float(np.mean(np.abs(out[5:-5, 0] - exact[5:-5, 0])))

    e1 = [error("upwind1", n) for n in (100, 200, 400, 800)]
    e2 = [error("tvd2", n) for n in (100, 200, 400, 800)]
    ratio1 = e1[0] / e1[-1]
    ratio2 = e2[0] / e2[-1]
    assert ratio2 > 2.0 * ratio1, (
        f"tvd2 should beat upwind1 by more than 2x at this resolution; "
        f"got upwind {ratio1:.2f}, tvd {ratio2:.2f}"
    )


@pytest.mark.parametrize("scheme", ["upwind1", "tvd2"])
def test_advection_integrates_to_the_boundary_flux_difference(scheme: str) -> None:
    """Both axial schemes conserve their inlet-minus-outlet flux exactly."""
    dz = 0.1
    velocity = 0.7
    c = np.array([[1.0], [0.4], [0.8], [0.2]])
    c_inlet = np.array([1.3])
    derivative = axial_advection(c, velocity, dz, c_inlet, scheme)
    np.testing.assert_allclose(
        dz * derivative.sum(axis=0),
        -velocity * (c[-1] - c_inlet),
        atol=1e-14,
    )


def test_tvd2_is_monotone_on_a_step() -> None:
    """No overshoot on a discontinuity. A stiff source will punish any."""
    n_z = 200
    dz = 1.0 / n_z
    z = np.arange(n_z) * dz
    c = np.where(z < 0.5, 0.0, 1.0)[:, None]
    dt = 0.5 * dz / 0.5
    for scheme in ("upwind1", "tvd2"):
        out = axial_advection(c, 0.5, dz, np.array([0.0]), scheme=scheme)
        c_next = c + dt * out
        assert c_next.min() >= -1e-14, f"{scheme} undershoots below zero"
        assert c_next.max() <= 1.0 + 1e-14, f"{scheme} overshoots above one"


def test_axial_dispersion_is_zero_when_switched_off() -> None:
    c = np.linspace(1.0, 0.0, 12)[:, None]
    assert np.max(np.abs(axial_dispersion(c, 0.0, 0.1))) == 0.0


def test_axial_dispersion_is_exact_for_a_quadratic() -> None:
    """Second-order stencil, so a quadratic must be differentiated exactly."""
    n_z = 9
    dz = 1.0 / n_z
    z = np.arange(n_z) * dz
    c = (1.0 - 2.0 * z + 3.0 * z**2)[:, None]
    out = axial_dispersion(c, 0.01, dz)
    interior = slice(1, -1)
    assert np.max(np.abs(out[interior, 0] - 0.01 * 6.0)) < 1e-12


# --- closures --------------------------------------------------------------


def test_no_external_resistance_returns_the_bulk_value() -> None:
    closure = NoExternalResistance()
    out = closure.shell_concentration(np.array([5.0]), np.array([1.0]), 1e-5, 1e-5)
    assert out[0] == 5.0


def test_film_resistance_limits_are_correct() -> None:
    """`k_g -> inf` must give pseudo-homogeneity, `k_g -> 0` must reflect."""
    bulk = np.array([2.0])
    inner = np.array([1.0])
    d_eff, dr = 1.0e-5, 1.0e-5

    fast = FilmResistance(k_g=1.0e3).shell_concentration(bulk, inner, d_eff, dr)
    assert fast[0] == pytest.approx(2.0, rel=2e-3)

    slow = FilmResistance(k_g=1.0e-9).shell_concentration(bulk, inner, d_eff, dr)
    assert slow[0] == pytest.approx(1.0, rel=1e-3)

    # In between, the shell sits between inner and bulk and is monotone in k_g.
    previous = 0.0
    for k_g in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        value = FilmResistance(k_g).shell_concentration(bulk, inner, d_eff, dr)[0]
        assert value > previous
        assert 1.0 <= value <= 2.0
        previous = value


def test_external_specific_area_includes_the_solid_fraction() -> None:
    """`a_v = 3(1 - eps_b)/R_p`, and the `(1 - eps_b)` must be there."""
    radius = 100e-6
    a_full = external_specific_area(radius, 0.0)
    a_half = external_specific_area(radius, 0.5)
    assert a_full == pytest.approx(3.0 / radius)
    assert a_half == pytest.approx(1.5 / radius)


def test_dispersion_closure_refuses_an_unphysical_peclet() -> None:
    assert isinstance(dispersion_for(0.0, 0.1, 1.0), NoDispersion)
    assert isinstance(dispersion_for(1e-3, 0.1, 1.0), DanckwertsDispersion)
    with pytest.raises(ValueError, match="Pe="):
        dispersion_for(1.0, 0.1, 1.0)


def test_uniform_pellet_has_no_radial_states() -> None:
    assert Uniform().carries_radial_states is False
    assert Uniform().n_nodes == 1
    assert ExplicitDiffusion(n_radial=12).carries_radial_states is True
    assert ExplicitDiffusion(n_radial=12).n_nodes == 12


def test_algebraic_surface_carries_no_states_and_relaxation_does() -> None:
    assert AlgebraicEquilibrium().carries_surface_states is False
    assert Relaxation(tau=1e-2).carries_surface_states is True
    assert np.all(AlgebraicEquilibrium().derivative(np.ones(3), np.zeros(3)) == 0.0)


def test_relaxation_derivative_points_toward_the_target() -> None:
    closure = Relaxation(tau=0.5)
    rate = closure.derivative(np.array([0.0]), np.array([1.0]))
    assert rate[0] == pytest.approx(2.0)
    rate = closure.derivative(np.array([1.0]), np.array([0.0]))
    assert rate[0] == pytest.approx(-2.0)


def test_relaxation_rejects_a_nonpositive_time_constant() -> None:
    with pytest.raises(ValueError, match="tau"):
        Relaxation(tau=0.0)


# --- sparsity --------------------------------------------------------------


def test_sparsity_shape_matches_the_state_vector() -> None:
    n_z, n_gas, n_radial, n_surface = 7, 4, 5, 3
    pattern = jacobian_sparsity(n_z, n_gas, n_radial, n_surface)
    expected = n_z * n_gas + n_z * n_radial * n_gas + n_z * n_radial * n_surface
    assert pattern.shape == (expected, expected)


def test_sparsity_covers_the_tvd2_axial_reconstruction_stencil() -> None:
    """Gas rows include the two-upwind/one-downwind TVD2 flux stencil.

    If this ever fails the sparsity pattern is wrong, and a wrong pattern makes
    the numerical Jacobian wrong -- a silent error that costs time rather than
    correctness, which is the worst kind to have.
    """
    n_z, n_gas, n_radial = 8, 4, 3
    pattern = jacobian_sparsity(n_z, n_gas, n_radial, None).tocsr()
    gas_size = n_z * n_gas
    for z in range(n_z):
        row = pattern.getrow(z * n_gas).toarray().ravel()
        non_zero = np.flatnonzero(row)
        gas_cols = non_zero[non_zero < gas_size]
        if gas_cols.size:
            assert gas_cols.min() >= max(0, z - 2) * n_gas
            assert gas_cols.max() < min(n_z, z + 2) * n_gas

    # At z=2, the left-face reconstruction uses the slope from z=1, which
    # depends on z=0 as its second-upwind concentration.
    assert pattern[2 * n_gas, 0] != 0


def test_sparsity_without_optional_blocks() -> None:
    plain = jacobian_sparsity(5, 4, None, None)
    assert plain.shape == (20, 20)
