# BEP-reactorresponsnotebook

Open [`bep_dynamic_reactor_response.ipynb`](bep_dynamic_reactor_response.ipynb) from the repository root. The notebook uses the existing `pbr` package, runs the transient integrations, embeds the resulting plots, and saves PNG copies under [`figures/`](figures/).

Included figures: [inlet/outlet](figures/inlet_outlet.png), [axial profiles](figures/axial_profiles.png), [radial profiles](figures/radial_profiles.png), [radial heatmaps](figures/radial_heatmaps.png), [last cycles](figures/last_cycles.png), and [R/D comparison](figures/radius_diffusivity_comparison.png).

## Starten

From the `Methanol-Catalysis` repository root:

```bash
pixi install
pixi run jupyter lab notebooks/bep_dynamic_reactor_response.ipynb
```

Select the `Python 3` kernel and choose **Run All Cells**. The first configuration cell contains the main controls for axial and radial resolution, forcing frequency and amplitude, number of periods, settling tolerance, and solver tolerances. The notebook also varies `R` and `D_e` in a small labelled sensitivity comparison.

Run the repository checks from the root with:

```bash
pixi run test
```

## Model scope

The example uses the `urmes2019` kinetic/operating set with the catalogued reactor geometry and an explicit finite-volume radial pellet. Surface coverages use the existing algebraic quasi-steady closure; the pellet surface concentration equals the bulk concentration, so no external-film resistance is included. The notebook shows the provenance of the geometry and transport inputs, including assumptions transferred from other alumina systems. Its figures are exploratory calculations and are not an experimental validation of the BEP.
