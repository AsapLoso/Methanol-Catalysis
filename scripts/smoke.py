"""Import check: the package must load and the default model must assemble.

Run with `pixi run python scripts/smoke.py`. Kept as a script rather than only
as a test because it is the fastest possible answer to "is the thing wired
together at all", and it prints the numbers a reader wants to see first.
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pbr
from pbr import (
    ExplicitDiffusion,
    FilmResistance,
    Inputs,
    NoExternalResistance,
    Relaxation,
    Uniform,
    build_model,
    make_inlet,
    zero_state,
)
from pbr.chemistry import rate_law
from pbr.parameter_sets import (
    borodzinski_2000,
)
from pbr.solver import atom_balance


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def main() -> int:
    banner("package")
    print(f"pbr {pbr.__version__}")

    banner("catalogue")
    for name in ("urmes2019", "borodzinski2000", "larsen2024"):
        inputs = Inputs.catalogued(name)
        model = build_model(inputs, make_inlet(inputs.operating))
        law = model.rate
        y = zero_state(model)
        c_bulk, _, _ = model.layout.unpack(y)
        rates = law.rates(
            model.inlet(0.0)[None, :] * 8.314462618 * model.temperature,
            model.temperature,
        )
        print(f"{name:18s} states={model.layout.size:6d}  T={model.temperature:7.2f} K")
        print(f"{'':18s} {law.name}")
        print(f"{'':18s} net H2 rate = {rates.gas[0, 0]:.6e} mol/(kg_cat s)")
        atoms = atom_balance(y, model.layout)
        print(f"{'':18s} atoms at t=0: C={atoms[0]:.3e} H={atoms[1]:.3e}")
        assert c_bulk.shape == (model.layout.n_z, 4)

    banner("closures change the state size")
    base = Inputs.catalogued("urmes2019")
    variants = {
        "all defaults (pellet resolved)": base,
        "well-mixed pellet": Inputs(
            geometry=base.geometry,
            catalyst=base.catalyst,
            operating=base.operating,
            kinetics=base.kinetics,
            solver=base.solver,
            radial=Uniform(),
            external=NoExternalResistance(),
            surface=base.surface,
        ),
        "film + relaxation": Inputs(
            geometry=base.geometry,
            catalyst=base.catalyst,
            operating=base.operating,
            kinetics=base.kinetics,
            solver=base.solver,
            radial=ExplicitDiffusion(n_radial=12, geometry=2),
            external=FilmResistance(k_g=1.0e-2),
            surface=Relaxation(tau=1.0e-2),
        ),
    }
    for label, inputs in variants.items():
        model = build_model(inputs, make_inlet(inputs.operating))
        print(f"{label:34s} -> {model.layout.size:6d} states")

    banner("rate laws reject an unsupportable combination")
    try:
        rate_law(borodzinski_2000(), site_basis=0.0)
    except ValueError as exc:
        print(f"borodzinski2000 with no metal loading -> refused: {exc}")

    print("\nsmoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
