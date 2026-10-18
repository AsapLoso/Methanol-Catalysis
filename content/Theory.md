---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Theory

The model describes the transient behaviour of a fixed-bed catalytic reactor in which
acetylene is selectively semi-hydrogenated to ethylene. This chapter states the physical
and chemical formulation: the reaction network, the conservation laws, the boundary
conditions, and the dimensionless groups that govern the regime.

## 1. The reaction network

The chemistry is modelled with a Langmuir–Hinshelwood–Hougen–Watson (LHHW) mechanism. Five
elementary steps compete for surface sites:

1. **Dissociative adsorption of hydrogen**
   $$\mathrm{H_2 + 2* \rightleftharpoons 2H*} \qquad R_1 = k_{1,\mathrm{ads}} P_{\mathrm{H_2}} \theta_*^2 - k_{1,\mathrm{des}} \theta_{\mathrm{H}}^2$$

2. **Molecular adsorption of acetylene**
   $$\mathrm{C_2H_2 + * \rightleftharpoons C_2H_2*} \qquad R_2 = k_{2,\mathrm{ads}} P_{\mathrm{C_2H_2}} \theta_* - k_{2,\mathrm{des}} \theta_{\mathrm{C_2H_2}}$$

3. **Surface hydrogenation to ethylene (reversible)**
   $$\mathrm{C_2H_2* + 2H* \rightleftharpoons C_2H_4* + 2*} \qquad R_3 = k_{3,\mathrm{fwd}} \theta_{\mathrm{C_2H_2}} \theta_{\mathrm{H}}^2 - k_{3,\mathrm{rev}} \theta_{\mathrm{C_2H_4}} \theta_*^2$$

4. **Desorption of the target product**
   $$\mathrm{C_2H_4* \rightleftharpoons C_2H_4 + *} \qquad R_4 = k_{4,\mathrm{des}} \theta_{\mathrm{C_2H_4}} - k_{4,\mathrm{ads}} P_{\mathrm{C_2H_4}} \theta_*$$

5. **Over-hydrogenation to ethane (irreversible, direct to gas phase)**
   $$\mathrm{C_2H_4* + 2H* \rightarrow C_2H_6(g) + 3*} \qquad R_5 = k_{5} \theta_{\mathrm{C_2H_4}} \theta_{\mathrm{H}}^2$$

The vacant-site fraction closes the site balance:
$$\theta_* = 1 - \theta_{\mathrm{H}} - \theta_{\mathrm{C_2H_2}} - \theta_{\mathrm{C_2H_4}}$$

**Why these five steps and not more.** Ethane is deliberately absent from the adsorbed set.
It leaves the surface immediately: there is no adsorbed ethane intermediate in the literature
this model is parameterised from, and inventing one would be an unsourced mechanistic claim.
The species set is chosen so that an element balance is checkable — the cheapest test that a
stoichiometry table has been transcribed correctly.

**Why macro rate laws.** Every rate law in the model returns a rate directly from partial
pressures and temperature, with no adsorbed intermediate appearing explicitly. That is what
the available data supports: the parameter database holds *composite* rate constants — products
of elementary constants that the source chose to report as a single number — so `dtheta/dt`
cannot be reconstructed from them. Refusing to invent the missing elementary constants costs
mechanistic detail and buys a model whose every rate is traceable to a table in a published
paper.

**The one exception.** The default kinetic set reports its adsorption terms individually rather
than as products. Its coverages can therefore be written down, and with relaxation surface
dynamics it becomes a genuine transient model — which is why it is the default. The other two
parameterisations are algebraic-equilibrium only.

**Non-monotonicity is preserved deliberately.** All three laws are monotone in hydrogen and
*decrease* in acetylene over part of their range — the rate maximum the sources both report.
A model in which the rate rises monotonically in its own reactant cannot exhibit the
competitive-adsorption regime that makes periodic operation worth studying.

## 2. Conservation laws

The reactor is described by three coupled balances on a void-volume basis, following
{cite:t}`bergerDynamicMethodsCatalytic2008` Eq. (1) and Eq. (2).

**Bulk gas** (along the bed, coordinate $z$):

$$\frac{\partial C_{b,i}}{\partial t} = -\frac{u_s}{\varepsilon_b}\frac{\partial C_{b,i}}{\partial z} + \frac{D_{\mathrm{ax}}}{\varepsilon_b}\frac{\partial^2 C_{b,i}}{\partial z^2} + \frac{\rho_s (1-\varepsilon_b)}{\varepsilon_b}\langle R_i \rangle_{\mathrm{particle}}$$

**Intra-particle gas** (radially, coordinate $r$, geometry exponent $s$: 0 slab, 1 cylinder, 2 sphere):

$$\frac{\partial c_{p,i}}{\partial t} = \frac{D_e}{\varepsilon_p}\frac{1}{r^s}\frac{\partial}{\partial r}\left(r^s \frac{\partial c_{p,i}}{\partial r}\right) + \frac{\rho_s}{\varepsilon_p} R_i$$

**Surface coverages** (relaxation toward quasi-steady state, time constant $\tau$):

$$\frac{\partial \theta_j}{\partial t} = \frac{\theta_j^{\mathrm{ss}} - \theta_j}{\tau}$$

The bulk balance carries four terms: accumulation, convective transport, axial dispersion, and
catalytic generation/consumption. Axial dispersion is negligible under the plug-flow criterion
$Pe_{\mathrm{ax}} > 100$, and is off by default.

**Three conventions that had to be pinned down.** Each is a silent factor of $\varepsilon_p$ or
$\varepsilon_b$ if you get it wrong:

- $u_s$ is the **superficial** velocity, based on the empty tube cross-section. The interstitial
  velocity is $u_s/\varepsilon_b$, and interstitial gas is what advection transports — so the
  transport term carries $u_s/\varepsilon_b$. Writing a bare $u_s$ against a bulk-volume
  concentration is the same physics with the void fraction folded into the concentration; the
  two forms cannot be mixed without introducing exactly one spurious $\varepsilon_b$.
- $D_e$ is **porosity-inclusive**, $D_e = \varepsilon_p D_m / \tau$, matching the transport
  reference, and is therefore divided by $\varepsilon_p$ in the pellet balance to give
  diffusivity per unit pore volume. Defined instead as $D_m/\tau$, that division would have to
  disappear. Both conventions circulate; they differ by exactly $\varepsilon_p$.
- The rate law is normalised to mol/(kg_cat·s), so the geometric prefactor carries the density:
  $\rho_s/\varepsilon_p$ converts it to mol/(m³_pore·s) inside the pellet, and
  $\rho_s(1-\varepsilon_b)/\varepsilon_b$ to mol/(m³_void·s) in the bulk. The two are consistent
  by construction — the particle equation volume-averaged over the pellet and multiplied by the
  bed solid fraction reproduces the bulk term exactly.

## 3. Boundary conditions

With dispersion switched off, the axial problem is **hyperbolic**, and the outflow condition is
zero-gradient rather than Danckwerts. This is a consequence of the plug-flow assumption, not an
independent modelling choice: imposing a Danckwerts outflow on a hyperbolic problem
over-specifies it.

At the pellet centre the radial operator is singular at $r = 0$ and the limit must be taken
before discretising. With symmetry $f'(0) = 0$, l'Hôpital resolves the $0/0$ to give a centre
coefficient of $2(1+s)$ rather than a tabulated constant — see the Methods chapter.

## 4. The four closures

A packed-bed model is a stack of approximations. Each is isolated as an object with a small
interface, so switching it is a change of argument rather than an edit inside the right-hand
side. The design goal is that the right-hand side reads as a transcription of the governing
equations, with no branches in it.

| Closure | Question it answers | Default here |
|---|---|---|
| `ExternalTransport` | Does gas cross a film to reach the pellet? | **No** — the measured Carberry number is ~3e-4 against a 0.05 threshold, so the film is inactive by a factor of 150 |
| `RadialTransport` | Does concentration vary inside a pellet? | **Yes, resolved** — not a convenience approximation: it is the reason the pellet is resolved at all. Switching it off gives the well-mixed-pellet limit |
| `SurfaceDynamics` | Does the catalyst remember? | **Algebraic equilibrium by default** — it is what the steady-state literature is calibrated under, and it is also the assumption the transient study is testing. `Relaxation` is one argument away |
| `AxialDispersion` | Does the gas back-mix? | **Off** — a packed bed at these flow rates is plug flow |

Reducing every closure to its simplest setting gives a plug-flow, well-mixed-pellet,
memoryless-catalyst, purely convective reactor. Turning them all on gives a dispersed reactor
with a resolved pellet and a relaxing surface. Same right-hand side either way.

## 5. Dimensionless groups

The regime is characterised by the Péclet number for axial dispersion
($Pe_{\mathrm{ax}} = u_s L / D_{\mathrm{ax}}$, plug flow above ~100), the Damköhler number
comparing reaction rate to convective transport, and the particle-scale Thiele modulus
comparing intrinsic reaction rate to intra-particle diffusion. The Carberry number, quoted
above for the film-resistance decision, compares observed to maximum external mass-transfer
rate.

## 6. What the theory does not claim

The model does not attempt to reproduce the surface mechanism from first principles — the
composite rate constants forbid it. It does not include an adsorbed ethane intermediate. It
does not treat the catalyst as deactivating. Each of these is a deliberate boundary on the
claims the model can support, and each is stated here rather than left implicit.
