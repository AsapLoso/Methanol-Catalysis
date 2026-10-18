---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# The reactor, physically

A fixed-bed reactor is a tube packed with porous catalyst pellets. Gas flows through the voids
between the pellets; the reaction happens on the internal surface of the pellets, which the gas
must reach by diffusing into the pores.

Three length scales matter, and the model resolves all three:

| Scale | What happens there | How the model treats it |
|---|---|---|
| **Bed** (cm–m) | Gas flows along the tube; concentration changes with position and time | Axial coordinate $z$, $N_z$ cells |
| **Pellet** (mm) | Gas diffuses radially inward while reacting | Radial coordinate $r$, $N_r$ nodes per cell |
| **Surface** (nm) | Molecules adsorb, react, and desorb on active sites | Fractional coverages $\theta_j$ |

The separation of these scales is what makes the problem stiff: surface processes equilibrate
in microseconds to milliseconds, while the gas composition along the bed evolves over seconds.

## Why periodic operation is interesting

Under steady feed, the catalyst settles into a fixed surface composition. The selectivity to
ethylene is then set by the balance between the desired partial hydrogenation and the undesired
full hydrogenation to ethane — a balance that cannot be improved by changing a constant
temperature or pressure alone.

Forcing the inlet composition to vary in time breaks that equilibrium. Because the two reaction
pathways have different dependencies on the surface hydrogen coverage, a periodic inlet can
drive the surface through states that a steady feed never reaches. The open question is which
input profiles do this best — and that question requires a dynamic model to answer, which is
what this work provides.
