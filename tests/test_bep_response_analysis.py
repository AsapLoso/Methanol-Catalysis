"""Checks for the BEP notebook's phase-aligned cycle analysis."""

from __future__ import annotations

import numpy as np
import pytest

from notebooks.bep_response_analysis import (
    analyse_settled_cycles,
    plot_parameter_comparison,
    plot_radial_heatmaps,
)


def test_settling_finds_a_converged_suffix_and_summarises_full_cycles() -> None:
    phase = np.linspace(0.0, 1.0, 101)
    reference = np.column_stack(
        [
            1.0 + 0.2 * np.sin(2.0 * np.pi * phase),
            0.3 + 0.05 * np.cos(2.0 * np.pi * phase),
        ]
    )
    amplitudes = [0.55, 0.75, 0.92, 1.0, 1.0, 1.0]
    outlet_by_cycle = np.stack([amplitude * reference for amplitude in amplitudes])

    result = analyse_settled_cycles(
        outlet_by_cycle,
        phase,
        ("C2H2", "C2H4"),
        relative_tolerance=0.01,
        required_consecutive_pairs=2,
        periods_to_summarise=3,
    )

    assert result.settled
    assert result.converged_pairs_at_end == 2
    assert result.first_settled_cycle == 3
    assert [row["cycle"] for row in result.summary_cycles] == [4, 5, 6]
    np.testing.assert_allclose(
        result.summary_cycles[-1]["mean"]["C2H2"], 1.0, atol=1e-4
    )


def test_settling_does_not_claim_a_converged_suffix_for_drifting_cycles() -> None:
    phase = np.linspace(0.0, 1.0, 51)
    waveform = 1.0 + 0.25 * np.sin(2.0 * np.pi * phase)
    amplitudes = np.linspace(0.5, 1.5, 8)
    outlet_by_cycle = amplitudes[:, None, None] * waveform[None, :, None]

    result = analyse_settled_cycles(
        outlet_by_cycle,
        phase,
        ("C2H2",),
        relative_tolerance=0.001,
        required_consecutive_pairs=2,
    )

    assert not result.settled
    assert result.first_settled_cycle is None
    assert result.summary_cycles == ()


def test_parameter_comparison_plot_accepts_one_curve_per_case() -> None:
    import matplotlib

    matplotlib.use("Agg")
    phase = np.linspace(0.0, 0.01, 11)
    values = np.ones((2, phase.size, 4))

    figure, axes = plot_parameter_comparison(
        phase, values, ("H2", "C2H2", "C2H4", "C2H6"), ("basis", "R×2, D×4")
    )

    assert len(axes.flat) == 4
    assert len(axes.flat[0].lines) == 2
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_radial_heatmap_reports_bad_rank_as_value_error() -> None:
    with pytest.raises(ValueError, match=r"shape \(phase, axial, radius, species\)"):
        plot_radial_heatmaps(
            np.linspace(0.0, 1.0, 3),
            np.linspace(0.1, 0.9, 2),
            np.ones(3),
            0,
            ("H2", "C2H2", "C2H4", "C2H6"),
        )


def test_settling_rejects_nonmonotone_phase_grid() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        analyse_settled_cycles(
            np.ones((2, 3, 1)),
            np.array([0.0, 1.0, 0.5]),
            ("C2H2",),
        )
