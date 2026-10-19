# BEP-reactorresponsnotebook

Open [`bep_dynamic_reactor_response.ipynb`](bep_dynamic_reactor_response.ipynb) from the repository root. The notebook uses the existing `pbr` package, runs the transient integrations, embeds the resulting plots, and saves PNG copies under [`figures/`](figures/).

Included figures: [inlet/outlet](figures/inlet_outlet.png), [axial profiles](figures/axial_profiles.png), [radial profiles with the surface boundary](figures/radial_profiles.png), [radial heatmaps](figures/radial_heatmaps.png), [last cycles](figures/last_cycles.png), [period-pair settling diagnostic](figures/settling_diagnostics.png), and [R/D comparison](figures/radius_diffusivity_comparison.png).

## Starten

From the `Methanol-Catalysis` repository root:

```bash
pixi install
pixi run jupyter lab notebooks/bep_dynamic_reactor_response.ipynb
```

Select the `Python 3` kernel and choose **Run All Cells**. The configuration cell exposes pellet radius and porosity, optional temperature/pressure/flow overrides, optional species-specific `D_e` overrides, axial and radial resolution, inlet forcing, number of periods, settling tolerance, and solver tolerances. A `None` operating override keeps the selected catalogue value; a `None` diffusivity override uses the existing Fuller correlation.

Manual `D_e` values must be supplied for all four gases and have a traceable source or measurement. The default `75 µm` radius, `0.50` pellet porosity, particle density, tortuosity, and parts of the reactor geometry remain assumptions carried by the catalogue.

The R/D comparison integrates three radius/diffusivity pairs with constant `R²/D_e` for every gas (`R = 37.5, 75, 150 µm`) plus one control with a changed ratio. It reports the full outlet-trajectory differences and both `R²/D_e` and `ε_p R²/D_e`. These are implementation-similarity checks, not experimental validation.

The last-period section compares the five final adjacent cycle pairs with a configurable relative-RMS threshold. The threshold is explicitly a numerical heuristic. Cycle means are shown for all gases only when the required consecutive suffix passes.

Run the repository checks from the root with:

```bash
pixi run test
```

## Model scope

The example uses the `urmes2019` PSSA rate law with the catalogued reactor and operating inputs and a cell-centred finite-volume radial pellet. Surface coverages use the existing algebraic quasi-steady closure; the physical surface value equals the local bulk concentration, so no external film resistance is included. The notebook and tests verify conservative bulk/pore exchange, reaction inventory accounting, Fuller units, finite-difference Jacobian coverage, sparse/dense BDF agreement, and constant-`R²/D_e` scaling. They do not establish source-matched kinetics, measured BEP pellet properties, dynamic surface kinetics, mesh convergence, or experimental validation.
