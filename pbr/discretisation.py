"""Discretisation: operators, weights, and the Jacobian sparsity pattern.

Three things live here, and they are here together because they constrain each
other: the radial operator, the axial operator, and the sparsity pattern that
tells the integrator how the two couple.

The radial operator
-------------------
Written in flux form, `(1/r^s) d/dr (r^s dc/dr)`, discretised across cell
faces at `r_{i±1/2}`. Flux form rather than the equivalent
`d^2c/dr^2 + (s/r) dc/dr` because flux form has one stencil instead of two,
so the two terms cannot drift out of the discrete conservation the continuous
operator has.

The centre node is where this gets interesting. The operator is singular at
`r = 0` and the limit has to be taken before discretising. Doing it properly:

    (1/r^s) d/dr (r^s f')  ->  f'' + (s/r) f'

As `r -> 0` with symmetry `f'(0) = 0`, the second term's `0/0` resolves
against the first via l'Hopital to `(s-1) f''(0)`, giving
`(1+s) f''(0)`. The second derivative at the centre comes from the ghost node
at `r = -dr`, and symmetry says its value equals the node at `+dr`. So

    f''(0) = (c_1 - 2 c_0 + c_{-1}) / dr^2 = 2 (c_1 - c_0) / dr^2

and the full centre operator is

    2 (1+s) (c_1 - c_0) / dr^2          ->  6 (c_1 - c_0) / dr^2  for s = 2

The coefficient `2(1+s)` is the one to carry, not the `6`. Writing `6` hardcoded
gives the right answer for spheres and silently the wrong one for cylinders
(`4`, where `2(1+1)` also gives `4`) and slabs (`2`). Keeping the `s` in the
expression costs nothing and makes the code correct for all three geometries,
which is why it is derived rather than tabulated.

Axial operator
--------------
Backward-difference upwind by default. Central differences would be second
order and would ring on the square-wave inlet this model is meant to handle,
producing concentrations below zero -- and a stiff chemical source amplifies a
negative concentration into a visible artefact immediately. Monotonicity beats
order here. `TVD2` is available for smooth transients where the extra order is
worth the limiter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pbr.closures import ExplicitDiffusion


@dataclass(frozen=True, slots=True)
class RadialGrid:
    """Node positions, face positions, and volume weights for a pellet.

    Built once and handed to the operator, so a bug in the weights cannot
    silently disagree with a bug in the stencil: they come from the same
    object.
    """

    r_p: float
    n_nodes: int
    geometry: int

    @property
    def s(self) -> int:
        """0 slab, 1 cylinder, 2 sphere."""
        return self.geometry

    @property
    def r(self) -> NDArray[np.float64]:
        return np.linspace(0.0, self.r_p, self.n_nodes)

    @property
    def dr(self) -> float:
        if self.n_nodes < 2:
            raise ValueError("need at least two nodes for a finite spacing")
        return self.r_p / (self.n_nodes - 1)

    @property
    def r_half(self) -> NDArray[np.float64]:
        """Cell-face radii, length `n_nodes + 1`, from `-dr/2` to `R + dr/2`.

        One more face than nodes, because each of the `n_nodes` nodes is
        surrounded by two faces. Getting this length wrong makes the
        divergence below broadcast against the wrong shape, which is at least a
        loud failure.
        """
        dr = self.dr
        return np.arange(-1, self.n_nodes) * dr + dr / 2.0

    @property
    def weights(self) -> NDArray[np.float64]:
        """Quadrature weights for the volume average of a radial quantity.

        Composite Simpson where the node count allows it, trapezoid otherwise.
        Simpson needs an even number of intervals, so it needs an *odd* number
        of nodes -- which is why the default `n_radial` is 25 and not 24. It
        matters because the integrand is `r^s` times the rate, and `r^s` is a
        cubic at worst for a sphere. Trapezoid is exact only for linear
        integrands, so for `s = 2` it carries an `O(dr^2)` error in every
        particle-averaged rate; Simpson is exact for that integrand and the
        error drops to round-off.

        Uniform weights are wrong here for a geometric reason: there are
        `n_nodes` nodes but only `n_nodes - 1` intervals, so giving every node a
        full `dr` over-counts the domain by one cell's worth -- a factor
        `n/(n-1)`, i.e. 25% at `N_r = 5`.

        Normalising by the discrete sum rather than by the analytic integral is
        what closes this out. An earlier scaffold summed a left-endpoint rule
        and divided by the analytic value, which made the weights sum to 0.84
        and biased every particle-averaged rate ~16% low. Normalising by the
        discrete sum makes the weights integrate a constant to exactly one for
        any grid, so that whole class of bias is structurally impossible.
        """
        n = self.n_nodes
        if n == 1:
            return np.ones(1)
        dr = self.dr
        if n % 2 == 1:
            weights = np.ones(n)
            weights[1:-1:2] = 4.0
            weights[2:-1:2] = 2.0
            weights *= dr / 3.0
        else:
            weights = np.full(n, dr)
            weights[0] = dr / 2.0
            weights[-1] = dr / 2.0
        volume_weight = weights * self.r**self.s
        total = volume_weight.sum()
        if total <= 0.0:
            raise ValueError(
                "degenerate quadrature: all weights vanish. A single radial "
                "node at r=0 has zero volume."
            )
        return volume_weight / total

    @property
    def volume(self) -> float:
        """Exact pellet volume [m^3] for the geometry."""
        if self.s == 2:
            return 4.0 / 3.0 * np.pi * self.r_p**3
        if self.s == 1:
            return np.pi * self.r_p**2 * 1.0  # unit length cylinder
        return 2.0 * self.r_p  # unit area slab


def centre_operator(s: int, dr: float, diffusivity: float) -> float:
    """Coefficient of `(c_1 - c_0)` in the centre-node diffusion operator.

    `2 (1+s) D_e / (eps_p dr^2)`. Derived in the module docstring. Exposed
    because it is the single number most often got wrong in pellet models:
    the value `6` without the `2 (1+s)` structure works for spheres and is
    wrong everywhere else.
    """
    return 2.0 * (1.0 + s) * diffusivity / dr**2


def radial_diffusion(
    c: NDArray[np.float64],
    grid: RadialGrid,
    d_eff: float,
    porosity: float,
    shell: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """`(D_e/eps_p) * (1/r^s) d/dr (r^s dc/dr)` at every node.

    `c` has shape (..., n_nodes); `shell`, when given, replaces the value at
    the outermost node and is how a Robin condition enters. It is an *input*,
    not something this function writes back -- the earlier scaffolding version
    assigned into the state vector, which silently corrupted the solver's own
    trajectory. Everything here is read-only in `c`.
    """
    if c.shape[-1] != grid.n_nodes:
        raise ValueError(f"expected {grid.n_nodes} radial nodes, got {c.shape[-1]}")

    if shell is not None:
        if shell.ndim < c.ndim:
            shell = np.expand_dims(shell, axis=-1)
        c = np.concatenate([c[..., :-1], shell], axis=-1)

    s = grid.s
    dr = grid.dr
    scale = d_eff / porosity

    # Interior face fluxes at r_{i+1/2} for i = 0 .. n_nodes - 2
    r_int = (np.arange(grid.n_nodes - 1) + 0.5) * dr
    flux_int = r_int**s * (c[..., 1:] - c[..., :-1]) / dr

    # Symmetry at centre face gives zero flux; outer face flux is zero for closed/Dirichlet boundary
    f_0 = np.zeros((*c.shape[:-1], 1), dtype=c.dtype)
    f_N = np.zeros((*c.shape[:-1], 1), dtype=c.dtype)
    face_flux = np.concatenate([f_0, flux_int, f_N], axis=-1)

    r_mid = np.maximum(grid.r, 1e-14)
    divergence = np.diff(face_flux, axis=-1) / (r_mid**s * dr)
    result = scale * divergence

    # The divergence above has a 0/0 limit at r = 0. Replace it with the exact centre limit.
    centre = centre_operator(s, dr, d_eff) / porosity * (c[..., 1] - c[..., 0])
    return np.concatenate([centre[..., None], result[..., 1:]], axis=-1)


def axial_advection(
    c: NDArray[np.float64],
    u_s: float,
    dz: float,
    c_inlet: NDArray[np.float64],
    scheme: str = "upwind1",
) -> NDArray[np.float64]:
    """`-u_s dc/dz` on a uniform grid, cell-centred with the inlet as a ghost.

    `c` is (n_z, n_species). The inlet enters as a ghost value at `z = -dz/2`,
    which makes the first cell see exactly the same stencil as every other one
    -- no special-casing, no off-by-one, and the Dirichlet condition satisfied
    to second order.
    """
    n_z = c.shape[0]
    padded = np.concatenate([c_inlet[None, :], c], axis=0)

    if scheme == "upwind1":
        derivative = (padded[1:] - padded[:-1]) / dz
    elif scheme == "tvd2":
        derivative = _tvd2_derivative(padded, dz)
    else:
        raise ValueError(f"unknown advection scheme {scheme!r}")

    return -u_s * derivative[:n_z]


def _tvd2_derivative(padded: NDArray[np.float64], dz: float) -> NDArray[np.float64]:
    """Monotonized Central (MC) limited second-order derivative, over `n_z = len(padded) - 1` cells.

    Cell `i` has a backward difference to cell `i-1` and a forward difference to cell `i+1`.
    The MC limiter preserves second-order accuracy on smooth regions while remaining strictly
    monotone across discontinuities.
    """
    backward = (padded[1:] - padded[:-1]) / dz
    forward = np.empty_like(backward)
    forward[:-1] = (padded[2:] - padded[1:-1]) / dz
    forward[-1] = backward[-1]  # outflow: zero-gradient, no forward neighbour

    central = 0.5 * (backward + forward)
    two_b = 2.0 * backward
    two_f = 2.0 * forward
    cond = backward * forward > 0.0
    mag = np.minimum(np.abs(central), np.minimum(np.abs(two_b), np.abs(two_f)))
    return np.where(cond, np.sign(central) * mag, backward)


def axial_dispersion(
    c: NDArray[np.float64],
    d_ax: float,
    dz: float,
) -> NDArray[np.float64]:
    """`D_ax d^2c/dz^2` with homogeneous Neumann ends.

    Only used when dispersion is switched on; the ghost nodes mirror the
    boundary values, which is the correct closure for a no-flux wall.
    """
    if d_ax <= 0.0:
        return np.zeros_like(c)
    padded = np.concatenate([c[:1], c, c[-1:]], axis=0)
    second = (padded[2:] - 2.0 * padded[1:-1] + padded[:-2]) / dz**2
    return d_ax * second


def jacobian_sparsity(
    n_z: int,
    n_gas: int,
    n_radial: int | None,
    n_surface: int | None,
) -> csr_matrix:
    """Analytic sparsity pattern of `dr/dy`.

    Every coupling is nearest-neighbour in `z` and in `r`, and there is no
    coupling between different species. So the matrix is block-banded with
    bandwidth `n_gas` in both directions.

    Supplying this is worth roughly two orders of magnitude on a resolved
    pellet. SciPy's default is a dense finite-difference Jacobian, which costs
    one full right-hand-side evaluation *per state variable* -- 1480 of them
    for a small pellet. Declaring the pattern lets it evaluate only the
    neighbours, which for this structure is a small constant times `N_z`
    rather than times `N_z * N_gas * N_r`.

    The pattern is deliberately conservative: where unsure, include the entry.
    A pattern that is too sparse makes the numerical Jacobian wrong, which is a
    silent wrong answer, whereas too dense only costs time.
    """
    gas_size = n_z * n_gas
    radial_size = 0 if n_radial is None else n_z * n_radial * n_gas
    surface_size = 0 if n_surface is None else n_z * (n_radial or 1) * n_surface
    total = gas_size + radial_size + surface_size

    rows: list[int] = []
    cols: list[int] = []

    def _add(row: int, col: int) -> None:
        rows.append(row)
        cols.append(col)

    # Bulk gas. Advection couples to the two neighbouring axial cells; the
    # reaction couples the cell to every radial node in it.
    for z in range(n_z):
        for i in range(n_gas):
            row = z * n_gas + i
            for zz in range(max(0, z - 1), min(n_z, z + 2)):
                for j in range(n_gas):
                    _add(row, zz * n_gas + j)
            if n_radial:
                for r in range(n_radial):
                    _add(row, gas_size + (z * n_radial + r) * n_gas + i)

    # Particle: radial coupling only, within a column.
    if n_radial:
        for z in range(n_z):
            for r in range(n_radial):
                for i in range(n_gas):
                    row = gas_size + (z * n_radial + r) * n_gas + i
                    for rr in range(max(0, r - 1), min(n_radial, r + 2)):
                        _add(row, gas_size + (z * n_radial + rr) * n_gas + i)

    # Surface coverages couple radially, to the gas in the same cell, and to
    # each other -- the rate law is a dense function of the gas concentrations.
    if n_surface and n_radial:
        for z in range(n_z):
            for r in range(n_radial):
                for j in range(n_surface):
                    row = gas_size + radial_size + (z * n_radial + r) * n_surface + j
                    for rr in range(max(0, r - 1), min(n_radial, r + 2)):
                        for jj in range(n_surface):
                            _add(
                                row,
                                gas_size
                                + radial_size
                                + (z * n_radial + rr) * n_surface
                                + jj,
                            )
                    for i in range(n_gas):
                        _add(row, gas_size + (z * n_radial + r) * n_gas + i)

    data = np.ones(len(rows), dtype=np.int8)
    return csr_matrix((data, (rows, cols)), shape=(total, total))


def radial_grid_for(radial: ExplicitDiffusion | object, r_p: float) -> RadialGrid:
    """Build the grid for whatever radial closure was chosen."""
    n_nodes = radial.n_nodes
    geometry = getattr(radial, "geometry", 2)
    return RadialGrid(r_p=r_p, n_nodes=n_nodes, geometry=geometry)
