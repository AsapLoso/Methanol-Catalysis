# BEP reactor response notebook — V2 technical handoff

This note accompanies the V2 notebook in [Methanol-Catalysis PR #2](https://github.com/AsapLoso/Methanol-Catalysis/pull/2). V1 remains the snapshot in [PR #1](https://github.com/AsapLoso/Methanol-Catalysis/pull/1); V2 was branched from its live head. Both PRs remain open and unmerged.

## Run it

From the `Methanol-Catalysis` repository root:

```bash
pixi install
pixi run jupyter lab notebooks/bep_dynamic_reactor_response.ipynb
```

Choose the `Python 3` kernel, edit the configuration cell near the top, then run all cells in order. `None` for temperature, pressure, and flow keeps the selected `urmes2019` catalogue value. `None` for the diffusivity override uses the existing Fuller calculation. A manual `D_e` override must provide values for all four gases and should have a traceable source or measurement. The final cell records software versions and solver statistics; the plots are saved under [`figures/`](figures/).

### Default configuration

| Input | Default | Meaning |
|---|---:|---|
| Kinetic set | `urmes2019` | Existing catalogued PSSA rate law and operating set |
| Pellet radius | 75 µm | Catalogue-derived starting value |
| Pellet porosity | 0.50 | Explicit transferred assumption |
| Temperature / pressure / flow | catalogue values | Each can be overridden independently |
| Effective diffusivity | Fuller estimate | Optional mapping for H₂, C₂H₂, C₂H₄, and C₂H₆ |
| Axial / radial cells | 8 / 12 | Cell-centred axial bed and radial pellet resolution |
| Inlet forcing | 200 Hz, 10% | Cosine modulation of the tracked feed components |
| Duration / samples | 16 periods / 60 per period | 80 ms total at the default forcing frequency |
| Integrator / tolerances | BDF; `rtol=1e-4`, `atol=1e-8` | Change these when checking numerical sensitivity |
| Settling heuristic | 1% relative RMS; 5 adjacent pairs | Must pass for every tracked gas |

## Suggested review order

1. **[`inlet_outlet.png`](figures/inlet_outlet.png)** — dashed inlet and solid outlet histories for all four gases. This is the main reactor-response view.
2. **[`settling_diagnostics.png`](figures/settling_diagnostics.png), then [`last_cycles.png`](figures/last_cycles.png)** — first check phase-aligned error for adjacent periods against the 1% line; then inspect the final full outlet periods and the table of full-cycle means. The error threshold is a numerical heuristic, not evidence of physical or experimental validity.
3. **[`radial_profiles.png`](figures/radial_profiles.png) and [`radial_heatmaps.png`](figures/radial_heatmaps.png)** — inspect internal pellet gradients across selected phases and over the final period. Radial squares at `r/R=1` show the imposed physical surface value `c_p(R)=C_b`; interior curves are finite-volume cell values.
4. **[`axial_profiles.png`](figures/axial_profiles.png)** — compare bulk concentrations along the bed at selected phases of the last simulated period.
5. **[`radius_diffusivity_comparison.png`](figures/radius_diffusivity_comparison.png)** — compare complete last-period outlet histories for three constant-`R²/D_e` cases and a control whose ratio is four times larger.

## Numerical run receipt and findings

The checked V2 default run used Python 3.12.14, NumPy 2.5.3, SciPy 1.18.0, and Pydantic 2.13.5. The baseline solver succeeded at `N_z=8`, `N_r=12`, `R=75 µm`, and `ε_p=0.50`; its recorded integration time was about 0.46 s (`nfev=890`, `njev=4`, `nlu=80`). The notebook executed all 22 cells without errors and wrote seven PNG figures.

For the 200 Hz, 16-period run, the final ten adjacent period pairs pass the 1% criterion for all four gases; the consecutive passing suffix starts at period 6. Every one of the final five displayed pairs passes. Their largest component error is 0.02304% (C₂H₆, periods 14→15). The notebook reports full-period means and half peak-to-peak swings for each gas over periods 14–16. This demonstrates that these computed outlet traces changed little between those cycles under the selected discretization and tolerance.

At the default `R=75 µm`, the notebook reports these species-specific scales:

| Gas | `D_e` (m²/s) | `R²/D_e` (ms) | `ε_p R²/D_e` (ms) |
|---|---:|---:|---:|
| H₂ | 1.226×10⁻⁵ | 0.4586 | 0.2293 |
| C₂H₂ | 2.444×10⁻⁶ | 2.3017 | 1.1509 |
| C₂H₄ | 2.304×10⁻⁶ | 2.4414 | 1.2207 |
| C₂H₆ | 2.061×10⁻⁶ | 2.7288 | 1.3644 |

The three fully integrated variants `(R/R₀, D_e/D₀)=(0.5,0.25),(1,1),(2,4)` have identical full outlet trajectories in the recorded calculation. In the control `(2,1)`, which makes `R²/D_e` four times larger, the maximum relative full-trajectory differences versus baseline are H₂ 0.05352%, C₂H₂ 0.2062%, C₂H₄ 0.2810%, and C₂H₆ 1.650%. This is a similarity property of this implemented PSSA model and parameter scaling; it is not a claim that the radial state can be removed.

## What was tested

The V2 branch reports `63 passed` with `python -m pytest -q`, and Ruff passed with `uv tool run --offline ruff check pbr tests notebooks/bep_response_analysis.py`. The added regression coverage checks zero-reaction bulk/pore inventory exchange, one pellet reaction source and C/H balance, the no-film surface value and radial-face behavior, Fuller units and inert partial pressures, finite-difference coverage of the Jacobian sparsity mask, sparse-versus-dense stiff BDF trajectories, and all-species constant-`R²/D_e` integrations. These checks verify code-level numerical properties; they do not validate the model against an experiment.

Re-run the repository suite from the repository root with:

```bash
pixi run test
```

## Implemented model and limits

- Axial plug-flow transport with axial dispersion disabled; a spherical pellet represented by finite, cell-centred radial volumes.
- Zero flux at the pellet centre and the physical no-film Dirichlet boundary `c_p(R,t)=C_b(z,t)`. The bulk and pore balances use equal-and-opposite interfacial flux, and the reaction source is counted once in the pellet balance.
- The code's porosity-inclusive `D_e` convention is documented; the pore balance uses `D_e/ε_p`. Both `R²/D_e` and `ε_p R²/D_e` are shown.
- Surface coverages use the existing algebraic quasi-steady-state (PSSA) closure. No dynamic surface-coverage kinetics are implemented.
- The default Fuller diffusivities are estimates, not measurements for Damian's BEP pellet. The catalogue combines kinetics and operating values with reactor and pellet assumptions that are not fully source-matched to the Urmes 2019 experiment. Radius, porosity, tortuosity, particle density, and parts of the geometry therefore remain assumptions.
- The period-pair criterion does not establish mesh or time-step convergence. No calibration, detector/noise comparison, measured BEP response, or experimental validation is supplied. The radial fields are model states, not direct measurements.
- This handoff adds no new simulations or model experiments. The later pellet-population and reduced-order questions remain out of scope for this V2 delivery.

The latest scientific handoff on [PM issue #58](https://github.com/AsapLoso/project-management-agents/issues/58#issuecomment-6257428574), its [cross-review on PR #1](https://github.com/AsapLoso/Methanol-Catalysis/pull/1#issuecomment-6260001345), and the [Agent A/B role clarification](https://github.com/AsapLoso/project-management-agents/issues/58#issuecomment-6260236632) are linked for provenance. Agent A's separate [scientific development-history PR](https://github.com/AsapLoso/project-management-agents/pull/70) owns the central BEP state reconciliation; this handoff does not edit those state files.
