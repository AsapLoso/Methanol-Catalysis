"""Small post-processing helpers used by the BEP reactor-response notebook.

These functions consume states returned by :mod:`pbr`; they do not add or
change reactor physics.  Keeping cycle sampling and the settling diagnostic
here makes those analysis choices easy to inspect and test independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

GAS_COLORS = {
    "H2": "#31688e",
    "C2H2": "#d7782b",
    "C2H4": "#6c8f3f",
    "C2H6": "#b45578",
}


def set_plot_style() -> None:
    """Apply a restrained, consistent Matplotlib style for notebook figures."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#1f2937",
            "axes.titlecolor": "#111827",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "grid.color": "#d9dee7",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def plot_inlet_outlet(
    time: NDArray[np.float64],
    inlet: NDArray[np.float64],
    bulk: NDArray[np.float64],
    species_names: tuple[str, ...],
):
    """Plot the inlet and last-axial-cell concentration for each gas."""
    import matplotlib.pyplot as plt

    time = np.asarray(time, dtype=float).reshape(-1)
    inlet = np.asarray(inlet, dtype=float).reshape(time.size, -1)
    bulk = np.asarray(bulk, dtype=float).reshape(time.size, -1, len(species_names))
    if inlet.shape[1] != len(species_names):
        raise ValueError("inlet species dimension does not match species_names")
    if bulk.shape[2] != len(species_names):
        raise ValueError("bulk species dimension does not match species_names")

    figure, axes = plt.subplots(
        2, 2, figsize=(11.5, 6.8), sharex=True, constrained_layout=True
    )
    for index, (axis, species) in enumerate(zip(axes.flat, species_names, strict=True)):
        color = GAS_COLORS.get(species, "#334155")
        axis.plot(
            time,
            inlet[:, index],
            linestyle="--",
            color=color,
            linewidth=1.4,
            label="inlaat",
        )
        axis.plot(time, bulk[:, -1, index], color=color, linewidth=2.0, label="uitlaat")
        axis.set_title(species)
        axis.set_ylabel("concentratie (mol m⁻³)")
        axis.set_xlabel("tijd (s)")
        axis.grid(True, alpha=0.8)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle("Inlaat en reactoruitlaat per gas", fontsize=15, color="#111827")
    return figure, axes


def plot_axial_profiles(
    z: NDArray[np.float64],
    bulk_snapshots: NDArray[np.float64],
    species_names: tuple[str, ...],
    snapshot_labels: tuple[str, ...],
):
    """Plot axial bulk profiles at selected phases of the forcing cycle."""
    import matplotlib.pyplot as plt

    z = np.asarray(z, dtype=float)
    profiles = np.asarray(bulk_snapshots, dtype=float)
    expected = (len(snapshot_labels), z.size, len(species_names))
    if profiles.shape != expected:
        raise ValueError(f"bulk_snapshots must have shape {expected}")

    snapshot_colors = ("#334e68", "#d7782b", "#6c8f3f", "#b45578", "#7c3aed")
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 6.8), constrained_layout=True)
    for species_index, (axis, species) in enumerate(
        zip(axes.flat, species_names, strict=True)
    ):
        for snapshot_index, label in enumerate(snapshot_labels):
            axis.plot(
                z * 1e3,
                profiles[snapshot_index, :, species_index],
                color=snapshot_colors[snapshot_index % len(snapshot_colors)],
                linewidth=1.9,
                label=label,
            )
        axis.set_title(species)
        axis.set_ylabel("concentratie (mol m⁻³)")
        axis.set_xlabel("axiale positie z (mm)")
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(True, alpha=0.8)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle(
        "Axiale bulkprofielen op geselecteerde fasen", fontsize=15, color="#111827"
    )
    return figure, axes


def plot_radial_heatmaps(
    phase_time: NDArray[np.float64],
    radius_fraction: NDArray[np.float64],
    particle_last_cycle: NDArray[np.float64],
    axial_index: int,
    species_names: tuple[str, ...],
):
    """Plot last-cycle concentration heatmaps across pellet radius and phase."""
    import matplotlib.pyplot as plt

    phase_time = np.asarray(phase_time, dtype=float)
    radius_fraction = np.asarray(radius_fraction, dtype=float)
    particle = np.asarray(particle_last_cycle, dtype=float)
    if particle.ndim != 4:
        raise ValueError(
            "particle_last_cycle must have shape (phase, axial, radius, species)"
        )
    expected_dims = (
        phase_time.size,
        particle.shape[1],
        radius_fraction.size,
        len(species_names),
    )
    if particle.shape != expected_dims:
        raise ValueError(f"particle_last_cycle must have shape {expected_dims}")
    if not 0 <= axial_index < particle.shape[1]:
        raise IndexError("axial_index is outside the particle state")

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    for species_index, (axis, species) in enumerate(
        zip(axes.flat, species_names, strict=True)
    ):
        values = particle[:, axial_index, :, species_index].T
        mesh = axis.pcolormesh(
            phase_time,
            radius_fraction,
            values,
            shading="auto",
            cmap="viridis",
        )
        axis.set_title(species)
        axis.set_ylabel("genormaliseerde straal r/R")
        axis.set_xlabel("tijd binnen laatste periode (s)")
        colorbar = figure.colorbar(mesh, ax=axis, label="concentratie (mol m⁻³)")
        colorbar.formatter.set_useOffset(False)
        colorbar.update_ticks()
    figure.suptitle(
        "Radiale pelletconcentratie tijdens de laatste periode",
        fontsize=15,
        color="#111827",
    )
    return figure, axes


def plot_radial_profiles(
    radius_fraction: NDArray[np.float64],
    particle_snapshots: NDArray[np.float64],
    axial_index: int,
    species_names: tuple[str, ...],
    snapshot_labels: tuple[str, ...],
    *,
    surface_snapshots: NDArray[np.float64] | None = None,
):
    """Plot radial cell values and optionally the physical no-film surface."""
    import matplotlib.pyplot as plt

    radius_fraction = np.asarray(radius_fraction, dtype=float)
    snapshots = np.asarray(particle_snapshots, dtype=float)
    expected = (
        len(snapshot_labels),
        snapshots.shape[1] if snapshots.ndim == 4 else 0,
        radius_fraction.size,
        len(species_names),
    )
    if snapshots.ndim != 4 or snapshots.shape != expected:
        raise ValueError(f"particle_snapshots must have shape {expected}")
    if not 0 <= axial_index < snapshots.shape[1]:
        raise IndexError("axial_index is outside the particle state")
    surface = None
    if surface_snapshots is not None:
        surface = np.asarray(surface_snapshots, dtype=float)
        expected_surface = (len(snapshot_labels), len(species_names))
        if surface.shape != expected_surface:
            raise ValueError(f"surface_snapshots must have shape {expected_surface}")

    phase_colors = ("#334e68", "#d7782b", "#6c8f3f", "#b45578", "#7c3aed")
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 6.8), constrained_layout=True)
    for species_index, (axis, species) in enumerate(
        zip(axes.flat, species_names, strict=True)
    ):
        for phase_index, label in enumerate(snapshot_labels):
            axis.plot(
                radius_fraction,
                snapshots[phase_index, axial_index, :, species_index],
                marker="o",
                markersize=3,
                color=phase_colors[phase_index % len(phase_colors)],
                linewidth=1.8,
                label=label,
            )
            if surface is not None:
                axis.scatter(
                    [1.0],
                    [surface[phase_index, species_index]],
                    marker="s",
                    s=30,
                    color=phase_colors[phase_index % len(phase_colors)],
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=4,
                )
        axis.set_xlim(0.0, 1.04 if surface is not None else 1.0)
        axis.set_title(species)
        axis.set_ylabel("concentratie (mol m⁻³)")
        axis.set_xlabel(
            "r/R (celcentra; vierkant op r/R=1: cₚ(R)=Cᵦ)"
            if surface is not None
            else "genormaliseerde straal r/R (celcentra)"
        )
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(True, alpha=0.8)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle(
        f"Radiale pelletprofielen bij axiale cel {axial_index + 1}",
        fontsize=15,
        color="#111827",
    )
    return figure, axes


def plot_cycle_overlay(
    phase: NDArray[np.float64],
    outlet_by_cycle: NDArray[np.float64],
    species_names: tuple[str, ...],
    cycle_indices: tuple[int, ...],
):
    """Overlay selected complete outlet cycles at their common phase."""
    import matplotlib.pyplot as plt

    phase = np.asarray(phase, dtype=float)
    cycles = np.asarray(outlet_by_cycle, dtype=float)
    if cycles.ndim != 3 or cycles.shape[1:] != (phase.size, len(species_names)):
        raise ValueError("outlet_by_cycle dimensions do not match phase/species")
    if not cycle_indices:
        raise ValueError("cycle_indices must contain at least one period")
    if any(not 0 <= index < cycles.shape[0] for index in cycle_indices):
        raise IndexError("cycle index is outside the sampled periods")
    last_cycle = cycle_indices[-1]

    figure, axes = plt.subplots(
        2, 2, figsize=(11.5, 6.8), sharex=True, constrained_layout=True
    )
    for species_index, (axis, species) in enumerate(
        zip(axes.flat, species_names, strict=True)
    ):
        color = GAS_COLORS.get(species, "#334155")
        for cycle_index in cycle_indices:
            axis.plot(
                phase,
                cycles[cycle_index, :, species_index],
                color=color,
                alpha=0.45 if cycle_index != last_cycle else 1.0,
                linewidth=1.3 if cycle_index != last_cycle else 2.2,
                label=f"periode {cycle_index + 1}",
            )
        axis.set_title(species)
        axis.set_ylabel("uitlaatconcentratie (mol m⁻³)")
        axis.set_xlabel("tijd binnen periode (s)")
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(True, alpha=0.8)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle("Laatste volledige uitlaatperioden", fontsize=15, color="#111827")
    return figure, axes


def plot_settling_diagnostics(
    relative_rms_error: NDArray[np.float64],
    species_names: tuple[str, ...],
    relative_tolerance: float,
):
    """Plot phase-aligned cycle-pair errors against the heuristic threshold."""
    import matplotlib.pyplot as plt

    errors = np.asarray(relative_rms_error, dtype=float)
    if errors.ndim != 2 or errors.shape[1] != len(species_names):
        raise ValueError("relative_rms_error must have shape (cycle pair, species)")
    if errors.shape[0] < 1 or not np.all(np.isfinite(errors)):
        raise ValueError("relative_rms_error must contain finite cycle-pair values")
    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")

    figure, axis = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    pair_numbers = np.arange(1, errors.shape[0] + 1)
    for species_index, species in enumerate(species_names):
        plotted = np.maximum(errors[:, species_index], relative_tolerance * 1e-4)
        axis.semilogy(
            pair_numbers,
            plotted,
            marker="o",
            markersize=4,
            linewidth=1.7,
            color=GAS_COLORS.get(species, "#334155"),
            label=species,
        )
    axis.axhline(
        relative_tolerance,
        color="#9f1239",
        linestyle="--",
        linewidth=1.4,
        label=f"heuristische grens {100 * relative_tolerance:g}%",
    )
    axis.set_xlabel("opeenvolgend periodepaar (eerdere → volgende)")
    axis.set_ylabel("relatieve RMS-afwijking (–, log-schaal)")
    axis.set_title("Periode-op-periode verschil per uitlaatgas")
    axis.grid(True, which="both", alpha=0.55)
    axis.legend(ncol=2, fontsize=9)
    return figure, axis


def plot_parameter_comparison(
    phase: NDArray[np.float64],
    outlet_by_case: NDArray[np.float64],
    species_names: tuple[str, ...],
    case_labels: tuple[str, ...],
):
    """Compare phase-resolved outlet curves across R/D sensitivity cases."""
    import matplotlib.pyplot as plt

    phase = np.asarray(phase, dtype=float)
    cases = np.asarray(outlet_by_case, dtype=float)
    expected = (len(case_labels), phase.size, len(species_names))
    if cases.shape != expected:
        raise ValueError(f"outlet_by_case must have shape {expected}")
    if not case_labels:
        raise ValueError("case_labels must contain at least one case")

    case_colors = ("#1f4e79", "#d97706", "#2a9d8f", "#7c3aed")
    line_styles = ("-", "--", ":", "-.")
    figure, axes = plt.subplots(
        2, 2, figsize=(11.5, 6.8), sharex=True, constrained_layout=True
    )
    for species_index, (axis, species) in enumerate(
        zip(axes.flat, species_names, strict=True)
    ):
        for case_index, label in enumerate(case_labels):
            axis.plot(
                phase,
                cases[case_index, :, species_index],
                color=case_colors[case_index % len(case_colors)],
                linestyle=line_styles[case_index % len(line_styles)],
                linewidth=2.2 if case_index == 0 else 1.7,
                label=label,
            )
        axis.set_title(species)
        axis.set_ylabel("uitlaatconcentratie (mol m⁻³)")
        axis.set_xlabel("tijd binnen periode (s)")
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(True, alpha=0.8)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle(
        "Laatste-periode-uitlaatrespons bij R/D-varianten",
        fontsize=15,
        color="#111827",
    )
    return figure, axes


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Decoded pbr states on a requested time grid.

    Concentrations retain the shapes ``(time, axial, species)`` for bulk gas
    and ``(time, axial, radius, species)`` for pellet gas. ``time`` is 1D for
    a direct decode and ``(period, phase)`` for period-aligned samples. Optional
    state blocks are ``None`` when the selected pbr closure does not carry them.
    """

    time: NDArray[np.float64]
    inlet: NDArray[np.float64]
    bulk: NDArray[np.float64]
    particle: NDArray[np.float64] | None
    theta: NDArray[np.float64] | None


@dataclass(frozen=True, slots=True)
class SettlingAnalysis:
    """Result of a phase-aligned, consecutive-period comparison."""

    settled: bool
    relative_rms_error: NDArray[np.float64]
    converged_pairs_at_end: int
    first_settled_cycle: int | None
    summary_cycles: tuple[dict[str, Any], ...]


def decode_states(
    model: Any, times: NDArray[np.float64], states: NDArray[np.float64]
) -> Trajectory:
    """Decode a ``solve_ivp`` state matrix through pbr's public ``Layout``.

    Parameters
    ----------
    model:
        A built pbr model.
    times:
        One-dimensional sample times in seconds.
    states:
        State matrix in SciPy convention, ``(n_state, n_time)``.
    """
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if states.ndim != 2 or states.shape[1] != times.size:
        raise ValueError("states must have shape (n_state, len(times))")
    if states.shape[0] != model.layout.size:
        raise ValueError(
            f"state dimension {states.shape[0]} does not match pbr layout "
            f"({model.layout.size})"
        )

    bulk_values: list[NDArray[np.float64]] = []
    particle_values: list[NDArray[np.float64]] = []
    theta_values: list[NDArray[np.float64]] = []
    has_particle = model.layout.carries_radial
    has_theta = model.layout.carries_surface
    for state in states.T:
        bulk, particle, theta = model.layout.unpack(state)
        bulk_values.append(bulk)
        if has_particle:
            particle_values.append(particle)
        if has_theta:
            theta_values.append(theta)

    return Trajectory(
        time=times,
        inlet=np.stack([model.inlet(float(t)) for t in times]),
        bulk=np.stack(bulk_values),
        particle=np.stack(particle_values) if has_particle else None,
        theta=np.stack(theta_values) if has_theta else None,
    )


def sample_solution_by_period(
    model: Any,
    solution: Any,
    *,
    start_time: float,
    period: float,
    n_periods: int,
    samples_per_period: int,
) -> tuple[Trajectory, NDArray[np.float64]]:
    """Sample dense solver output on a phase-aligned grid of full periods.

    Returns the decoded trajectory and a ``(period, phase)`` time array.  The
    endpoint is included in every period so curves can be plotted cleanly;
    period averages use trapezoidal integration and therefore do not double
    count that endpoint.
    """
    if solution.sol is None:
        raise ValueError("integrate with dense_output=True before resampling")
    if period <= 0.0 or n_periods < 1 or samples_per_period < 2:
        raise ValueError("period, n_periods, and samples_per_period are invalid")

    phase = np.linspace(0.0, period, samples_per_period + 1)
    period_times = start_time + np.arange(n_periods)[:, None] * period + phase[None, :]
    flat_times = period_times.reshape(-1)
    sampled_states = solution.sol(flat_times)
    trajectory = decode_states(model, flat_times, sampled_states)

    def reshape_time_axis(values: NDArray[np.float64] | None):
        if values is None:
            return None
        return values.reshape((n_periods, samples_per_period + 1, *values.shape[1:]))

    cycle_trajectory = Trajectory(
        time=period_times,
        inlet=trajectory.inlet.reshape((n_periods, samples_per_period + 1, -1)),
        bulk=reshape_time_axis(trajectory.bulk),
        particle=reshape_time_axis(trajectory.particle),
        theta=reshape_time_axis(trajectory.theta),
    )
    return cycle_trajectory, phase


def analyse_settled_cycles(
    outlet_by_cycle: NDArray[np.float64],
    phase: NDArray[np.float64],
    species_names: tuple[str, ...],
    *,
    relative_tolerance: float = 0.01,
    required_consecutive_pairs: int = 3,
    periods_to_summarise: int = 3,
) -> SettlingAnalysis:
    """Find a settled suffix using relative RMS differences between cycles.

    Two cycles are compared at identical forcing phase.  A pair passes only
    when every species' RMS difference divided by the larger RMS magnitude of
    those two outlet curves is at most ``relative_tolerance``.  The notebook
    labels this an analysis heuristic, not a physical or experimental
    acceptance criterion.
    """
    cycles = np.asarray(outlet_by_cycle, dtype=float)
    phase = np.asarray(phase, dtype=float)
    if cycles.ndim != 3:
        raise ValueError("outlet_by_cycle must have shape (cycle, phase, species)")
    if cycles.shape[1] != phase.size:
        raise ValueError("phase length must match the cycle sample count")
    if cycles.shape[2] != len(species_names):
        raise ValueError("species_names length must match the species dimension")
    if cycles.shape[0] < 2:
        raise ValueError("at least two complete cycles are needed to assess settling")
    if phase.ndim != 1 or phase.size < 2 or np.any(np.diff(phase) <= 0.0):
        raise ValueError("phase must be one-dimensional and strictly increasing")
    if not np.all(np.isfinite(cycles)) or not np.all(np.isfinite(phase)):
        raise ValueError("cycle concentrations and phase must be finite")
    if not 0.0 <= relative_tolerance or required_consecutive_pairs < 1:
        raise ValueError(
            "settling tolerance and consecutive-pair count must be positive"
        )
    if periods_to_summarise < 1:
        raise ValueError("periods_to_summarise must be positive")

    differences = np.diff(cycles, axis=0)
    difference_rms = np.sqrt(np.mean(differences**2, axis=1))
    cycle_rms = np.sqrt(np.mean(cycles**2, axis=1))
    denominator = np.maximum(cycle_rms[:-1], cycle_rms[1:])
    denominator = np.maximum(denominator, np.finfo(float).tiny)
    relative_error = difference_rms / denominator
    pair_passes = np.all(relative_error <= relative_tolerance, axis=1)

    trailing = 0
    for passed in pair_passes[::-1]:
        if not passed:
            break
        trailing += 1
    settled = trailing >= required_consecutive_pairs
    first_settled_cycle = cycles.shape[0] - 1 - trailing if settled else None

    summary: list[dict[str, Any]] = []
    if settled:
        n_summary = min(periods_to_summarise, trailing + 1)
        start = cycles.shape[0] - n_summary
        phase_span = float(phase[-1] - phase[0])
        if phase_span <= 0.0:
            raise ValueError("phase must span a positive interval")
        for index in range(start, cycles.shape[0]):
            values = cycles[index]
            mean = np.trapezoid(values, x=phase, axis=0) / phase_span
            amplitude = 0.5 * (np.max(values, axis=0) - np.min(values, axis=0))
            summary.append(
                {
                    "cycle": index + 1,
                    "mean": {
                        name: float(mean[i]) for i, name in enumerate(species_names)
                    },
                    "half_peak_to_peak": {
                        name: float(amplitude[i])
                        for i, name in enumerate(species_names)
                    },
                }
            )

    return SettlingAnalysis(
        settled=settled,
        relative_rms_error=relative_error,
        converged_pairs_at_end=trailing,
        first_settled_cycle=first_settled_cycle,
        summary_cycles=tuple(summary),
    )
