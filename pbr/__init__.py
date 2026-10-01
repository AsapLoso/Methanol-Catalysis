"""A transient packed-bed model for selective acetylene semi-hydrogenation.

The package is layered, and the layering is the design:

    provenance   what may enter, and on whose authority
    species      the chemical alphabet
    parameters   the shape of every parameter set
    parameter_sets  the values, each with a citation
    chemistry    rate laws behind one interface
    closures     the four places where detail can be switched off
    discretisation  operators, weights, sparsity
    solver       the right-hand side and the integrator
    model        assembly, defaults, inlets

Nothing below `model` constructs anything, and nothing above `solver` knows
what an ODE is.

Quick start::

    from pbr import Inputs, build_model, make_inlet, steady_state_initial_condition

    inputs = Inputs.catalogued("urmes2019")
    model = build_model(inputs, make_inlet(inputs.operating))
    y0 = steady_state_initial_condition(model)
    solution = model.integrate(y0, (0.0, 10.0))
"""

from pbr.chemistry import RateLaw, RateResult
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
from pbr.discretisation import RadialGrid, centre_operator, jacobian_sparsity
from pbr.model import (
    Inputs,
    build_model,
    constant_inlet,
    effective_diffusivities,
    make_inlet,
    steady_state_initial_condition,
    zero_state,
)
from pbr.parameters import (
    Catalyst,
    Geometry,
    KineticSet,
    OperatingPoint,
    SolverSettings,
    TransportProperties,
)
from pbr.provenance import Provenance, Registry, Source
from pbr.solver import Layout, Model, atom_balance
from pbr.species import Adsorbate, Gas

__version__ = "0.1.0"

__all__ = [
    "Adsorbate",
    "AlgebraicEquilibrium",
    "Catalyst",
    "DanckwertsDispersion",
    "ExplicitDiffusion",
    "FilmResistance",
    "Gas",
    "Geometry",
    "Inputs",
    "KineticSet",
    "Layout",
    "Model",
    "NoDispersion",
    "NoExternalResistance",
    "OperatingPoint",
    "Provenance",
    "RadialGrid",
    "RateLaw",
    "RateResult",
    "Registry",
    "Relaxation",
    "SolverSettings",
    "Source",
    "TransportProperties",
    "Uniform",
    "__version__",
    "atom_balance",
    "build_model",
    "centre_operator",
    "constant_inlet",
    "dispersion_for",
    "effective_diffusivities",
    "external_specific_area",
    "jacobian_sparsity",
    "make_inlet",
    "steady_state_initial_condition",
    "zero_state",
]
