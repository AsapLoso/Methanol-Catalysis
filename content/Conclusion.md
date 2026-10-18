---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Conclusion

## What was built

A transient packed-bed reactor model for the selective semi-hydrogenation of acetylene to
ethylene, implemented as a small, tested Python package. The model couples bulk gas transport
along the bed, intra-particle diffusion in the radial direction, and a relaxing catalyst
surface, with the reaction chemistry supplied by one of three literature parameterisations
behind a single interface.

The package separates the governing equations from the approximations applied to them. Four
closures — external film resistance, intra-particle diffusion, surface dynamics, and axial
dispersion — are objects passed into the model rather than branches inside it. The right-hand
side reads as a transcription of the equations, which is what makes the model auditable from
the outside.

## What was learned

**The modelling decisions are the work.** The most valuable output of this phase is not the
code but the reasoning attached to it: why flux form rather than the equivalent two-term
radial operator; why the centre coefficient is $2(1+s)$ rather than a hardcoded $6$; why
monotonicity beats order on a square-wave inlet; why the default kinetic set is the one whose
adsorption terms are reported individually. Each of those is recorded where it was made, in
the module it affects, rather than in a design document that would drift.

**Refusing to invent is a design constraint, not a limitation.** The parameter literature
reports composite rate constants, so the elementary constants needed for a first-principles
surface mechanism are simply not available. The model refuses to synthesise them. The cost is
mechanistic detail; the benefit is that every rate in the model traces to a table in a
published paper.

**The interruptions were informative.** The work was started and repeatedly interrupted by bug
reports. Each interruption surfaced something the tests had not yet covered — which is the
argument for writing tests that assert *correctness of the discretisation* rather than merely
that the code runs. The suite now checks second-order convergence, exact volume weights, the
centre coefficient, monotonicity on a step, and the consistency between the particle and bulk
source terms.

## What is not done

- **No parameter estimation.** The recorded next step is to specify the reaction mechanism,
  stoichiometry matrix, kinetic rate laws, and literature-based operating and kinetic
  parameters, then simulate forward before any fitting is attempted.
- **No validation against experimental data.** The model is internally consistent — the tests
  check that — but it has not been compared against measured transient responses.
- **No optimal input-profile design.** The stated purpose is to use the model to design
  transient inlet profiles that improve ethylene selectivity. That control step has not begun.
- **Single pellet geometry per run.** The operator supports slab, cylinder and sphere, but the
  parameter sets are configured for one geometry at a time.

## Next steps

The immediate next step is to complete the parameter specification and run a forward
simulation. Parameter estimation follows only once the forward problem is trustworthy.
