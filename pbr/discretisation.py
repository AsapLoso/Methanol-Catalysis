"""Discretisation: operators, weights, and the Jacobian sparsity pattern.

Three things live here, and they are here together because they constrain each
other: the radial operator, the axial operator, and the sparsity pattern that
tells the integrator how the two couple.

Radial operators
----------------
The transient reactor model uses `FiniteVolumeRadialGrid` and
`radial_diffusion_fv`: cell-centred unknowns, zero flux at the symmetry plane,
and the physical pellet surface as a Dirichlet face. The operator is written
as a face-flux divergence, so its radial-volume integral is exactly the
surface transfer used with the opposite sign in the bulk balance. There is no
dynamic state at the surface.

`RadialGrid` and `radial_diffusion` retain the node-centred stencil used by
steady/reference operator checks. Its centre node needs a limiting argument:
the differential operator is singular at `r = 0`, so take the limit before
discretising. With symmetry `f'(0) = 0`:

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
The operators support first-order upwind and conservative TVD2 (Monotonized
Central face reconstruction). The catalogued model defaults to TVD2. A
centred derivative can ring near sharp inlet changes, producing negative
concentrations that a stiff chemical source can amplify. The flux-form TVD2
operator is second order in smooth regions and limits oscillations near steep
gradients; the time integration and reaction still require positivity checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass(frozen=True, slots=True)
class FiniteVolumeRadialGrid:
    """Cell-centred radial finite-volume grid with the pellet surface as a face.

    Unlike :class:`RadialGrid`, whose nodes include the centre and surface for
    profile and steady-state studies, these unknowns represent finite-volume
    cells.  The surface concentration is therefore a boundary value, not an
    extra dynamic state.  ``weights`` are the exact radial cell-volume
    fractions and ``r`` contains the cell-centre coordinates.
    """

    r_p: float
    n_nodes: int
    geometry: int
    _faces: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _centers: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _weights: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.r_p <= 0.0 or self.n_nodes < 1:
            raise ValueError("finite-volume radial grid needs R > 0 and N_r >= 1")
        if self.geometry not in (0, 1, 2):
            raise ValueError(
                "radial geometry must be 0 (slab), 1 (cylinder), or 2 (sphere)"
            )
        faces = np.linspace(0.0, self.r_p, self.n_nodes + 1)
        centers = (faces[:-1] + faces[1:]) / 2.0
        weights = (
            faces[1:] ** (self.s + 1) - faces[:-1] ** (self.s + 1)
        ) / self.r_p ** (self.s + 1)
        faces.setflags(write=False)
        centers.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "_faces", faces)
        object.__setattr__(self, "_centers", centers)
        object.__setattr__(self, "_weights", weights)

    @property
    def s(self) -> int:
        """0 slab, 1 cylinder, 2 sphere."""
        return self.geometry

    @property
    def dr(self) -> float:
        if self.n_nodes < 1:
            raise ValueError("need at least one radial cell")
        return self.r_p / self.n_nodes

    @property
    def r_half(self) -> NDArray[np.float64]:
        """Cell faces, including the symmetry plane and pellet surface."""
        return self._faces

    @property
    def r(self) -> NDArray[np.float64]:
        """Cell centres, from ``dr/2`` to ``R - dr/2``."""
        return self._centers

    @property
    def weights(self) -> NDArray[np.float64]:
        """Exact radial volume fractions for each finite-volume cell."""
        return self._weights

    @property
    def volume(self) -> float:
        """Exact pellet volume [m^3] for the geometry."""
        if self.s == 2:
            return 4.0 / 3.0 * np.pi * self.r_p**3
        if self.s == 1:
            return np.pi * self.r_p**2  # unit length cylinder
        return 2.0 * self.r_p  # unit area slab


def radial_diffusion_fv(
    c: NDArray[np.float64],
    grid: FiniteVolumeRadialGrid,
    d_eff: float,
    porosity: float,
    shell: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Conservative cell-centred ``(D_e/eps_p) * laplacian_r(c)``.

    ``shell`` is a Dirichlet value at ``r = R``.  The same surface flux used
    by the final pellet cell is returned by the volume integral of this
    operator, so pairing it with the opposite bulk transfer term conserves
    gas-phase inventory to round-off.
    """
    if c.shape[-1] != grid.n_nodes:
        raise ValueError(f"expected {grid.n_nodes} radial cells, got {c.shape[-1]}")
    if porosity <= 0.0:
        raise ValueError("porosity must be positive")

    shell = np.asarray(shell, dtype=float)
    if shell.ndim < c.ndim - 1:
        shell = np.expand_dims(shell, axis=-1)
    shell = np.broadcast_to(shell, c.shape[:-1])

    faces = grid.r_half
    centers = grid.r
    flux = np.zeros((*c.shape[:-1], grid.n_nodes + 1), dtype=float)
    if grid.n_nodes > 1:
        internal_gradient = np.diff(c, axis=-1) / np.diff(centers)
        flux[..., 1:-1] = faces[1:-1] ** grid.s * internal_gradient
    surface_distance = grid.r_p - centers[-1]
    surface_gradient = (shell - c[..., -1]) / surface_distance
    flux[..., -1] = grid.r_p**grid.s * surface_gradient

    radial_measure = (faces[1:] ** (grid.s + 1) - faces[:-1] ** (grid.s + 1)) / (
        grid.s + 1
    )
    return (d_eff / porosity) * np.diff(flux, axis=-1) / radial_measure


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
    """`-u_s dc/dz` on a uniform finite-volume grid with a fixed inlet flux."""
    n_z = c.shape[0]

    if scheme == "upwind1":
        padded = np.concatenate([c_inlet[None, :], c], axis=0)
        derivative = (padded[1:] - padded[:-1]) / dz
    elif scheme == "tvd2":
        derivative = _tvd2_flux_derivative(c, c_inlet, dz)
    else:
        raise ValueError(f"unknown advection scheme {scheme!r}")

    return -u_s * derivative[:n_z]


def _tvd2_flux_derivative(
    c: NDArray[np.float64], c_inlet: NDArray[np.float64], dz: float
) -> NDArray[np.float64]:
    """Conservative flux difference with Monotonized Central reconstruction.

    Values are finite-volume cell averages. The inlet fixes the flux at the
    left face and zero-gradient outflow uses the final cell value at the right
    face. Reconstructing the upwind state at internal faces keeps the update
    conservative while the MC limiter prevents step overshoot.
    """
    backward = np.empty_like(c)
    forward = np.empty_like(c)
    backward[0] = (c[0] - c_inlet) / (0.5 * dz)
    backward[1:] = np.diff(c, axis=0) / dz
    forward[:-1] = np.diff(c, axis=0) / dz
    forward[-1] = 0.0

    central = 0.5 * (backward + forward)
    two_b = 2.0 * backward
    two_f = 2.0 * forward
    cond = backward * forward > 0.0
    mag = np.minimum(np.abs(central), np.minimum(np.abs(two_b), np.abs(two_f)))
    slope = np.where(cond, np.sign(central) * mag, 0.0)

    face_flux = np.empty((c.shape[0] + 1, c.shape[1]), dtype=float)
    face_flux[0] = c_inlet
    if c.shape[0] > 1:
        face_flux[1:-1] = c[:-1] + 0.5 * dz * slope[:-1]
    face_flux[-1] = c[-1]
    return np.diff(face_flux, axis=0) / dz


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

    TVD2 axial flux reconstruction can couple a cell to two upwind cells and
    one downwind cell; radial transport couples neighbouring ``r`` cells. A
    kinetic rate in one cell depends on the full gas mixture there, so the
    local reaction block is dense across species. The resulting matrix remains
    block-banded in ``z`` and ``r``.

    Supplying this is worth roughly two orders of magnitude on a resolved
    pellet. SciPy's default is a dense finite-difference Jacobian, which costs
    one full right-hand-side evaluation *per state variable* -- 1480 of them
    for a small pellet. Declaring the pattern lets it evaluate only the
    local neighbours, which for this structure is a small constant times the
    number of axial and radial cells rather than the square of the state size.

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

    # Bulk gas. The conservative TVD2 reconstruction at a left face depends on
    # the upwind cell's slope, which can reach two cells upstream. Include the
    # second upwind block even when a run selects first-order upwind; that
    # conservative over-approximation keeps one pattern valid for both schemes.
    # A bulk source can depend on every gas species in the same radial column.
    for z in range(n_z):
        for i in range(n_gas):
            row = z * n_gas + i
            for zz in range(max(0, z - 2), min(n_z, z + 2)):
                for j in range(n_gas):
                    _add(row, zz * n_gas + j)
            if n_radial:
                for r in range(n_radial):
                    for j in range(n_gas):
                        _add(row, gas_size + (z * n_radial + r) * n_gas + j)

    # Particle: same-species radial diffusion plus local cross-species
    # kinetics. The outermost cell also sees its same-species bulk shell value.
    if n_radial:
        for z in range(n_z):
            for r in range(n_radial):
                for i in range(n_gas):
                    row = gas_size + (z * n_radial + r) * n_gas + i
                    for rr in range(max(0, r - 1), min(n_radial, r + 2)):
                        _add(row, gas_size + (z * n_radial + rr) * n_gas + i)
                    for j in range(n_gas):
                        _add(row, gas_size + (z * n_radial + r) * n_gas + j)
                    if r == n_radial - 1:
                        _add(row, z * n_gas + i)

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


def radial_grid_for(
    radial: ExplicitDiffusion | object, r_p: float
) -> FiniteVolumeRadialGrid:
    """Build the cell-centred finite-volume grid for a radial closure."""
    n_nodes = radial.n_nodes
    geometry = getattr(radial, "geometry", 2)
    return FiniteVolumeRadialGrid(r_p=r_p, n_nodes=n_nodes, geometry=geometry)
