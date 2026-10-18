---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Methods

The governing equations are continuous in space and time. This chapter states how they are
turned into a finite system that can be integrated: the spatial discretisation, the treatment
of the pellet centre, the choice of scheme, and the integrator.

The design principle throughout is that **the right-hand side should read as a transcription
of the governing equations**, with no branches in it. Every approximation is an object passed
in, not a conditional inside the derivative.

## 1. Method of Lines

The Method of Lines (MOL) discretises space and leaves time continuous. The partial differential
equations become a large system of ordinary differential equations in $t$, one per grid node,
which is then handed to a stiff integrator.

Two spatial coordinates are discretised:

- **Axial** — $N_z$ cells along the bed.
- **Radial** — $N_r$ nodes inside each pellet.

The state vector is therefore the bulk concentrations at every axial cell, the intra-particle
concentrations at every radial node of every cell, and the surface coverages. With the default
closures the axial problem stays hyperbolic and the pellet carries $N_r$ states per cell.

**Why MOL rather than a fully implicit space-time scheme.** MOL lets the spatial discretisation
and the time integration be reasoned about separately, and it means the stiff integrator sees an
explicit right-hand side it can evaluate and differentiate itself. The alternative — assembling
a large sparse implicit system by hand — buys nothing here and makes the physics harder to read
off the code.

## 2. The radial operator and the centre node

The radial operator is written in **flux form**,

$$\frac{1}{r^s}\frac{\partial}{\partial r}\left(r^s \frac{\partial c}{\partial r}\right)$$

discretised across cell faces at $r_{i\pm1/2}$. Flux form rather than the algebraically
equivalent $\frac{\partial^2 c}{\partial r^2} + \frac{s}{r}\frac{\partial c}{\partial r}$ because
flux form has **one stencil instead of two**, so the two terms cannot drift out of the discrete
conservation that the continuous operator satisfies.

### The centre is where this gets interesting

The operator is singular at $r = 0$, and the limit has to be taken *before* discretising. Doing
it properly:

$$\frac{1}{r^s}\frac{\partial}{\partial r}\left(r^s f'\right) \;\longrightarrow\; f'' + \frac{s}{r}f'$$

As $r \to 0$ with symmetry $f'(0) = 0$, the second term's $0/0$ resolves against the first via
l'Hôpital to $(s-1)f''(0)$, giving a total of $(1+s)f''(0)$. The second derivative at the centre
comes from the ghost node at $r = -dr$, and symmetry says its value equals the node at $+dr$:

$$f''(0) = \frac{c_1 - 2c_0 + c_{-1}}{dr^2} = \frac{2(c_1 - c_0)}{dr^2}$$

so the full centre operator is

$$\frac{2(1+s)(c_1 - c_0)}{dr^2} \;\longrightarrow\; \frac{6(c_1 - c_0)}{dr^2} \quad \text{for } s = 2$$

**The coefficient $2(1+s)$ is the one to carry, not the $6$.** Writing `6` hardcoded gives the
right answer for spheres and silently the wrong one for cylinders ($4$) and slabs ($2$). Keeping
the $s$ in the expression costs nothing and makes the code correct for all three geometries,
which is why it is *derived* rather than tabulated — and why there is a test asserting the
coefficient is exactly $2(1+s)$.

## 3. The axial scheme

Backward-difference upwind is the default. Central differences would be second order and would
**ring on the square-wave inlet** this model is meant to handle, producing concentrations below
zero — and a stiff chemical source amplifies a negative concentration into a visible artefact
immediately. Monotonicity beats order here.

`TVD2` is available for smooth transients, where the extra order is worth the limiter. It is
second order and monotone on a step, both of which are tested rather than asserted.

## 4. Volume weights

Node positions, face positions, and volume weights come from a single `RadialGrid` object, built
once and handed to the operator. This is deliberate: a bug in the weights cannot silently
disagree with a bug in the stencil if they come from the same object. The weights are exact for
the volume element and sum to one — including for even node counts, which is a case that is easy
to get wrong.

## 5. The integrator

The system is stiff, and the stiffness is physical rather than numerical: surface chemistry
relaxes on a far shorter timescale than convective transport. An explicit integrator would be
limited by the fastest surface timescale regardless of how slowly the bulk evolves.

The system is therefore integrated with an implicit multi-step / implicit Runge–Kutta solver
from `scipy.integrate`, which can take steps set by accuracy rather than by the fastest mode.
The Jacobian sparsity pattern is supplied: it is **block-tridiagonal in the axial direction**,
which the integrator uses to avoid dense factorisation. That pattern is itself tested.

## 6. Evaluation order

The right-hand side evaluates in a fixed order — bulk, shell, particle, chemistry, then
assembly — and the order matters. Each stage consumes the previous stage's output, so
reordering them silently changes the scheme from explicit to something else. It is stated in
the code rather than left implicit for exactly that reason.

## 7. What is verified, and how

The test suite does not assert that the code runs; it asserts that the discretisation is
*correct*. Representative checks:

- The first-order sphere and slab profiles match closed-form solutions.
- The sphere profile converges at **second order** under refinement.
- Volume weights are exact for the volume element and sum to one.
- The centre coefficient is exactly $2(1+s)$.
- Upwind advection is exact for a constant and has the right sign and magnitude.
- TVD2 is second order and upwind is not; TVD2 is monotone on a step.
- Axial dispersion is exactly zero when switched off, and exact for a quadratic when on.
- The sparsity pattern matches the state vector and is block-tridiagonal.
- The particle equation, volume-averaged and multiplied by the bed solid fraction, reproduces
  the bulk source term — the consistency check that the two geometric prefactors agree.
- The right-hand side does not mutate its input.

The last two are the ones that catch real bugs: a mutating right-hand side breaks an integrator
in ways that look like physics, and a mismatched prefactor produces a model that conserves
nothing while still running.
