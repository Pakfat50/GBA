"""ハイブリッド同定Stage 2: 非線形周期からIとKを決定する。

Stage 1で確定した頂点と波形別中心だけを入力にし、減衰係数は扱わない。
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fitting_tools import component_increments
from fitting_tools import ensure_directory
from fitting_tools import fit_quality_statistics
from fitting_tools import identify_base_inertia_and_restoring
from fitting_tools import nonlinear_period_samples
from fitting_tools import save_csv


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
DEFAULT_RESULT_ROOT = SCRIPT_DIRECTORY / "results"
CANDIDATE_MINIMUM_AMPLITUDE_DEG = 5.0
REFERENCE_AMPLITUDE_FRACTION = 0.10
REFERENCE_MINIMUM_PERIODS = 5
AMPLITUDE_BIN_WIDTH_DEG = 5.0
STABILITY_TOLERANCE_PERCENT = 1.0
STABILITY_QUANTILE = 0.80
STABILITY_CONSECUTIVE_BINS = 3
STABILITY_MINIMUM_BIN_SAMPLES = 10


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ハイブリッド同定Stage 2: 非線形周期によるI、K同定"
    )
    parser.add_argument("--date", required=True, help="試験日。例: 20260921")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def save_figure(figure, output_path, dpi=150):
    """完成前のPNGを読まれないよう、一時ファイルから原子的に置換する。"""

    temporary_path = output_path.with_name(
        output_path.stem + ".tmp" + output_path.suffix
    )
    figure.savefig(temporary_path, dpi=dpi)
    plt.close(figure)
    temporary_path.replace(output_path)


def read_inputs(parent_result):
    stage1_directory = parent_result / "hybrid_identification" / "01_preprocessing"
    turning_path = stage1_directory / "turning_points.csv"
    waveform_path = stage1_directory / "waveform_preprocessing.csv"
    manifest_path = parent_result / "resolved_manifest.csv"
    for path in [turning_path, waveform_path, manifest_path]:
        if not path.exists():
            raise FileNotFoundError("Stage 2の入力がありません: " + str(path))

    turning = pd.read_csv(turning_path)
    waveforms = pd.read_csv(waveform_path)
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required_turning = {
        "segment_id",
        "axis",
        "configuration",
        "direction",
        "peak_number",
        "peak_time_s",
        "centered_peak_angle_deg",
        "is_initial_peak",
    }
    required_waveforms = {
        "segment_id",
        "axis",
        "configuration",
        "direction",
        "repetition",
        "envelope_center_deg",
    }
    required_manifest = {
        "axis",
        "configuration",
        "spacer_count",
        "component_mass_kg",
        "signed_com_radius_m",
        "component_centroid_inertia_kg_m2",
        "force_lever_m",
        "use_for_calibration",
        "valid",
    }
    for name, table, required in [
        ("turning_points.csv", turning, required_turning),
        ("waveform_preprocessing.csv", waveforms, required_waveforms),
        ("resolved_manifest.csv", manifest, required_manifest),
    ]:
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(name + " に必要な列がありません: " + ", ".join(missing))
    return turning, waveforms, manifest, [turning_path, waveform_path, manifest_path]


def identify_waveform_frequencies(turning, waveforms):
    """各波形の同符号頂点周期と大振幅側の基準K/Iを求める。"""

    period_rows = []
    waveform_rows = []
    waveform_index = waveforms.set_index("segment_id")
    for segment_id, group in turning.groupby("segment_id", sort=False):
        group = group.sort_values("peak_number").reset_index(drop=True)
        initial_positions = np.flatnonzero(group["is_initial_peak"].to_numpy() == 1)
        if len(initial_positions) != 1:
            raise ValueError(segment_id + " の初期頂点が一意ではありません")
        group = group.iloc[int(initial_positions[0]) :].reset_index(drop=True)
        times = group["peak_time_s"].to_numpy(dtype=float)
        amplitudes = np.deg2rad(
            np.abs(group["centered_peak_angle_deg"].to_numpy(dtype=float))
        )
        samples = nonlinear_period_samples(
            times,
            amplitudes,
            CANDIDATE_MINIMUM_AMPLITUDE_DEG,
        )
        if len(samples) < 2:
            raise ValueError(segment_id + " に使用可能な周期が2個以上ありません")

        q_values = np.asarray(
            [sample["k_over_i_per_s2"] for sample in samples], dtype=float
        )
        sample_amplitudes = np.asarray(
            [sample["representative_amplitude_rad"] for sample in samples],
            dtype=float,
        )
        reference_count = max(
            REFERENCE_MINIMUM_PERIODS,
            int(math.ceil(REFERENCE_AMPLITUDE_FRACTION * len(samples))),
        )
        reference_count = min(reference_count, len(samples))
        reference_indices = np.argsort(sample_amplitudes)[-reference_count:]
        reference_index_set = {int(index) for index in reference_indices}
        reference_q = float(np.median(q_values[reference_indices]))
        metadata = waveform_index.loc[segment_id]
        for sample_number, sample in enumerate(samples):
            start_row = group.iloc[sample["start_peak_number"]]
            end_row = group.iloc[sample["end_peak_number"]]
            period_rows.append(
                {
                    "segment_id": segment_id,
                    "axis": str(metadata["axis"]),
                    "configuration": str(metadata["configuration"]),
                    "direction": str(metadata["direction"]),
                    "repetition": int(metadata["repetition"]),
                    "sample_number": sample_number,
                    "start_peak_number": int(start_row["peak_number"]),
                    "end_peak_number": int(end_row["peak_number"]),
                    "start_time_s": sample["start_time_s"],
                    "end_time_s": sample["end_time_s"],
                    "period_s": sample["period_s"],
                    "start_amplitude_deg": float(
                        np.rad2deg(sample["start_amplitude_rad"])
                    ),
                    "end_amplitude_deg": float(
                        np.rad2deg(sample["end_amplitude_rad"])
                    ),
                    "representative_amplitude_deg": float(
                        np.rad2deg(sample["representative_amplitude_rad"])
                    ),
                    "elliptic_parameter": sample["elliptic_parameter"],
                    "k_over_i_per_s2": sample["k_over_i_per_s2"],
                    "small_angle_k_over_i_per_s2": sample[
                        "small_angle_k_over_i_per_s2"
                    ],
                    "finite_amplitude_correction_ratio": sample[
                        "finite_amplitude_correction_ratio"
                    ],
                    "high_amplitude_reference_k_over_i_per_s2": reference_q,
                    "used_for_high_amplitude_reference": int(
                        sample_number in reference_index_set
                    ),
                    "relative_deviation_from_high_amplitude_reference_percent": float(
                        100.0
                        * (sample["k_over_i_per_s2"] - reference_q)
                        / reference_q
                    ),
                }
            )

        waveform_rows.append(
            {
                "segment_id": segment_id,
                "axis": str(metadata["axis"]),
                "configuration": str(metadata["configuration"]),
                "direction": str(metadata["direction"]),
                "repetition": int(metadata["repetition"]),
                "center_deg": float(metadata["envelope_center_deg"]),
                "candidate_periods": len(samples),
                "high_amplitude_reference_periods": reference_count,
                "high_amplitude_reference_k_over_i_per_s2": reference_q,
            }
        )
    return period_rows, waveform_rows


def derive_stable_period_selection(period_rows):
    """大振幅基準からの偏差率で安定振幅域と採用周期を決める。"""

    table = pd.DataFrame(period_rows)
    table["amplitude_bin_lower_deg"] = (
        np.floor(table["representative_amplitude_deg"] / AMPLITUDE_BIN_WIDTH_DEG)
        * AMPLITUDE_BIN_WIDTH_DEG
    )
    stability_rows = []
    axis_cutoffs = {}
    for axis_name in sorted(table["axis"].unique()):
        axis_table = table[table["axis"] == axis_name]
        for amplitude_bin, group in axis_table.groupby("amplitude_bin_lower_deg"):
            absolute_deviation = group[
                "relative_deviation_from_high_amplitude_reference_percent"
            ].abs()
            stability_rows.append(
                {
                    "axis": axis_name,
                    "amplitude_bin_lower_deg": float(amplitude_bin),
                    "amplitude_bin_upper_deg": float(
                        amplitude_bin + AMPLITUDE_BIN_WIDTH_DEG
                    ),
                    "number_of_period_samples": len(group),
                    "absolute_deviation_quantile_percent": float(
                        absolute_deviation.quantile(STABILITY_QUANTILE)
                    ),
                    "fraction_within_tolerance": float(
                        np.mean(
                            absolute_deviation.to_numpy()
                            <= STABILITY_TOLERANCE_PERCENT
                        )
                    ),
                }
            )

        axis_stability = sorted(
            [row for row in stability_rows if row["axis"] == axis_name],
            key=lambda row: row["amplitude_bin_lower_deg"],
        )
        cutoff = None
        for start in range(len(axis_stability) - STABILITY_CONSECUTIVE_BINS + 1):
            window = axis_stability[start : start + STABILITY_CONSECUTIVE_BINS]
            expected_bins = [
                window[0]["amplitude_bin_lower_deg"]
                + offset * AMPLITUDE_BIN_WIDTH_DEG
                for offset in range(STABILITY_CONSECUTIVE_BINS)
            ]
            actual_bins = [row["amplitude_bin_lower_deg"] for row in window]
            if actual_bins != expected_bins:
                continue
            enough_samples = all(
                row["number_of_period_samples"] >= STABILITY_MINIMUM_BIN_SAMPLES
                for row in window
            )
            stable = all(
                row["absolute_deviation_quantile_percent"]
                <= STABILITY_TOLERANCE_PERCENT
                for row in window
            )
            if enough_samples and stable:
                cutoff = window[0]["amplitude_bin_lower_deg"]
                break
        if cutoff is None:
            raise ValueError(axis_name + " 軸で安定振幅域を決定できません")
        axis_cutoffs[axis_name] = float(cutoff)

    common_cutoff = float(max(axis_cutoffs.values()))
    stability_lookup = {
        (row["axis"], row["amplitude_bin_lower_deg"]): row
        for row in stability_rows
    }
    for row in period_rows:
        amplitude_bin = float(
            np.floor(row["representative_amplitude_deg"] / AMPLITUDE_BIN_WIDTH_DEG)
            * AMPLITUDE_BIN_WIDTH_DEG
        )
        row["amplitude_bin_lower_deg"] = amplitude_bin
        row["axis_stable_amplitude_cutoff_deg"] = axis_cutoffs[row["axis"]]
        row["adopted_common_amplitude_cutoff_deg"] = common_cutoff
        row["within_common_stable_amplitude_range"] = int(
            row["representative_amplitude_deg"] >= common_cutoff
        )
        row["within_reference_deviation_tolerance"] = int(
            abs(row["relative_deviation_from_high_amplitude_reference_percent"])
            <= STABILITY_TOLERANCE_PERCENT
        )
        row["used_for_identification"] = int(
            row["within_common_stable_amplitude_range"] == 1
            and row["within_reference_deviation_tolerance"] == 1
        )
        if row["used_for_identification"] == 1:
            reason = "used"
        elif row["within_common_stable_amplitude_range"] == 0:
            reason = "below data-derived stable amplitude range"
        else:
            reason = "outside high-amplitude reference tolerance"
        row["selection_result"] = reason
        stability_lookup[(row["axis"], amplitude_bin)][
            "axis_stable_amplitude_cutoff_deg"
        ] = axis_cutoffs[row["axis"]]
        stability_lookup[(row["axis"], amplitude_bin)][
            "adopted_common_amplitude_cutoff_deg"
        ] = common_cutoff
    return period_rows, stability_rows, axis_cutoffs, common_cutoff


def finalize_waveform_frequencies(period_rows, waveform_rows):
    """採用周期だけから波形別K/I代表値を確定する。"""

    periods = pd.DataFrame(period_rows)
    final_rows = []
    for waveform in waveform_rows:
        segment_id = waveform["segment_id"]
        all_samples = periods[periods["segment_id"] == segment_id]
        used = all_samples[all_samples["used_for_identification"] == 1]
        if len(used) < 3:
            raise ValueError(segment_id + " の採用周期が3個未満です")
        q_values = used["k_over_i_per_s2"].to_numpy(dtype=float)
        q_median = float(np.median(q_values))
        q_mad = float(np.median(np.abs(q_values - q_median)))
        output = dict(waveform)
        output.update(
            {
                "periods_used": len(used),
                "periods_excluded": len(all_samples) - len(used),
                "period_mean_s": float(used["period_s"].mean()),
                "k_over_i_per_s2": q_median,
                "k_over_i_sample_mean_per_s2": float(np.mean(q_values)),
                "k_over_i_sample_std_per_s2": float(np.std(q_values, ddof=1)),
                "k_over_i_sample_median_absolute_deviation_per_s2": q_mad,
                "minimum_representative_amplitude_deg": float(
                    used["representative_amplitude_deg"].min()
                ),
                "maximum_representative_amplitude_deg": float(
                    used["representative_amplitude_deg"].max()
                ),
            }
        )
        final_rows.append(output)

        mask = (
            pd.Series([row["segment_id"] for row in period_rows]) == segment_id
        )
        for index in np.flatnonzero(mask.to_numpy()):
            period_rows[index]["adopted_waveform_k_over_i_per_s2"] = q_median
            period_rows[index][
                "relative_deviation_from_adopted_waveform_value_percent"
            ] = float(
                100.0
                * (period_rows[index]["k_over_i_per_s2"] - q_median)
                / q_median
            )
    return period_rows, final_rows


def summarize_configurations(waveform_rows, manifest):
    """波形代表K/Iを軸・形態ごとに集約し、既知増分を付ける。"""

    waveforms = pd.DataFrame(waveform_rows)
    rows = []
    valid_manifest = manifest[pd.to_numeric(manifest["valid"]) == 1]
    for unused_index, manifest_row in valid_manifest.iterrows():
        axis = str(manifest_row["axis"]).strip().upper()
        configuration = str(manifest_row["configuration"]).strip().upper()
        group = waveforms[
            (waveforms["axis"] == axis)
            & (waveforms["configuration"] == configuration)
        ]
        if len(group) == 0:
            continue
        values = group["k_over_i_per_s2"].to_numpy(dtype=float)
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        uncertainty = max(
            standard_deviation / np.sqrt(len(values)),
            0.001 * mean,
        )
        delta_inertia, delta_restoring = component_increments(manifest_row)
        rows.append(
            {
                "axis": axis,
                "configuration": configuration,
                "spacer_count": int(manifest_row["spacer_count"]),
                "use_for_calibration": int(manifest_row["use_for_calibration"]),
                "number_of_waveforms": len(values),
                "k_over_i_mean_per_s2": mean,
                "k_over_i_std_per_s2": standard_deviation,
                "k_over_i_uncertainty_per_s2": uncertainty,
                "delta_inertia_kg_m2": float(delta_inertia),
                "delta_restoring_n_m_per_rad": float(delta_restoring),
                "force_lever_m": float(manifest_row["force_lever_m"]),
            }
        )
    return rows


def identify_inertia_and_restoring(configuration_rows):
    """SP00～SP04から基準I,Kを分離し、全形態のI,Kを確定する。"""

    base_rows = []
    checked_configuration_rows = []
    parameter_rows = []
    axes = sorted({row["axis"] for row in configuration_rows})
    for axis in axes:
        axis_rows = [row for row in configuration_rows if row["axis"] == axis]
        calibration = []
        for row in axis_rows:
            if row["use_for_calibration"] != 1:
                continue
            calibration.append(
                {
                    **row,
                    "k_over_i_mean": row["k_over_i_mean_per_s2"],
                    "k_over_i_uncertainty": row["k_over_i_uncertainty_per_s2"],
                    "delta_inertia": row["delta_inertia_kg_m2"],
                    "delta_restoring": row["delta_restoring_n_m_per_rad"],
                }
            )
        base_inertia, base_restoring, checked = identify_base_inertia_and_restoring(
            calibration
        )
        measured = np.asarray([row["k_over_i_mean"] for row in checked])
        predicted = np.asarray([row["k_over_i_predicted"] for row in checked])
        rmse, correlation, r_squared = fit_quality_statistics(measured, predicted)
        base_rows.append(
            {
                "axis": axis,
                "base_inertia_kg_m2": base_inertia,
                "base_restoring_n_m_per_rad": base_restoring,
                "calibration_levels": len(calibration),
                "calibration_rmse_per_s2": rmse,
                "calibration_r": correlation,
                "calibration_r_squared": r_squared,
            }
        )
        checked_by_configuration = {
            row["configuration"]: row for row in checked
        }
        for row in axis_rows:
            measured_q = row["k_over_i_mean_per_s2"]
            gravity_inertia = base_inertia + row["delta_inertia_kg_m2"]
            gravity_restoring = (
                base_restoring + row["delta_restoring_n_m_per_rad"]
            )
            inertia = gravity_inertia
            restoring = gravity_restoring
            method = "base plus known spacer increments"
            if row["configuration"] == "BALL":
                inertia = gravity_restoring / measured_q
                method = "known gravity restoring divided by measured K/I"
            model_q = restoring / inertia
            q_residual = model_q - measured_q
            period_residual = (
                2.0 * np.pi / np.sqrt(model_q)
                - 2.0 * np.pi / np.sqrt(measured_q)
            )
            output = dict(row)
            output["k_over_i_predicted_per_s2"] = model_q
            output["k_over_i_residual_per_s2"] = q_residual
            output["k_over_i_relative_residual_percent"] = (
                100.0 * q_residual / measured_q
            )
            output["small_angle_period_residual_s"] = period_residual
            checked_configuration_rows.append(output)
            parameter_rows.append(
                {
                    "axis": axis,
                    "configuration": row["configuration"],
                    "inertia_kg_m2": float(inertia),
                    "geometric_inertia_kg_m2": float(gravity_inertia),
                    "restoring_n_m_per_rad": float(restoring),
                    "k_over_i_measured_per_s2": float(measured_q),
                    "k_over_i_model_per_s2": float(model_q),
                    "k_over_i_residual_per_s2": float(q_residual),
                    "small_angle_period_residual_s": float(period_residual),
                    "identification_method": method,
                }
            )
    return base_rows, checked_configuration_rows, parameter_rows


def compare_directions(waveform_rows):
    table = pd.DataFrame(waveform_rows)
    rows = []
    for (axis, configuration), group in table.groupby(["axis", "configuration"]):
        positive = group[group["direction"] == "P"]["k_over_i_per_s2"]
        negative = group[group["direction"] == "N"]["k_over_i_per_s2"]
        if len(positive) == 0 or len(negative) == 0:
            continue
        difference = float(positive.mean() - negative.mean())
        overall = float(group["k_over_i_per_s2"].mean())
        rows.append(
            {
                "axis": axis,
                "configuration": configuration,
                "positive_waveforms": len(positive),
                "negative_waveforms": len(negative),
                "positive_mean_k_over_i_per_s2": float(positive.mean()),
                "negative_mean_k_over_i_per_s2": float(negative.mean()),
                "positive_minus_negative_per_s2": difference,
                "relative_direction_difference_percent": 100.0 * difference / overall,
            }
        )
    return rows


def plot_frequency_calibration(configuration_rows, base_rows, output_path):
    table = pd.DataFrame(configuration_rows)
    base = pd.DataFrame(base_rows).set_index("axis")
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col")
    for column, axis_name in enumerate(["IN", "OUT"]):
        levels = table[
            (table["axis"] == axis_name)
            & (table["use_for_calibration"] == 1)
        ].sort_values("spacer_count")
        axes[0, column].errorbar(
            levels["spacer_count"],
            levels["k_over_i_mean_per_s2"],
            yerr=levels["k_over_i_uncertainty_per_s2"],
            fmt="o",
            capsize=4,
            label="Measured nonlinear-period K/I",
        )
        axes[0, column].plot(
            levels["spacer_count"],
            levels["k_over_i_predicted_per_s2"],
            "--",
            label="Shared I0, K0 calibration",
        )
        values = base.loc[axis_name]
        axes[0, column].set_title(
            f"{axis_name}: I0={values['base_inertia_kg_m2']:.6g}, "
            f"K0={values['base_restoring_n_m_per_rad']:.6g}, "
            f"R2={values['calibration_r_squared']:.6f}"
        )
        axes[0, column].set_ylabel("K/I [s$^{-2}$]")
        axes[0, column].legend(fontsize=8)
        axes[1, column].axhline(0.0, color="black", linewidth=0.8)
        axes[1, column].plot(
            levels["spacer_count"],
            levels["k_over_i_residual_per_s2"],
            "o-",
        )
        axes[1, column].set_xlabel("Spacer count")
        axes[1, column].set_ylabel("Model - measured K/I [s$^{-2}$]")
        for plot_axis in axes[:, column]:
            plot_axis.grid(True, alpha=0.25)
            plot_axis.set_xticks(levels["spacer_count"])
    figure.suptitle("Stage 2: inertia and restoring calibration from nonlinear periods")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_waveform_frequency(waveform_rows, configuration_rows, output_path):
    waveform = pd.DataFrame(waveform_rows).sort_values(
        ["axis", "configuration", "direction", "repetition"]
    ).reset_index(drop=True)
    configuration = pd.DataFrame(configuration_rows)
    configuration_order = ["BALL", "SP00", "SP01", "SP02", "SP03", "SP04"]
    colors = {
        name: plt.get_cmap("tab10")(index)
        for index, name in enumerate(configuration_order)
    }
    figure, axes = plt.subplots(2, 1, figsize=(22, 11), sharex=False)
    for plot_axis, axis_name in zip(axes, ["IN", "OUT"]):
        values = waveform[waveform["axis"] == axis_name].reset_index(drop=True)
        x = np.arange(len(values))
        for configuration_name in configuration_order:
            group = values[values["configuration"] == configuration_name]
            if len(group) == 0:
                continue
            positions = group.index.to_numpy()
            mean = configuration[
                (configuration["axis"] == axis_name)
                & (configuration["configuration"] == configuration_name)
            ]["k_over_i_mean_per_s2"].iloc[0]
            plot_axis.errorbar(
                positions,
                group["k_over_i_per_s2"],
                yerr=group["k_over_i_sample_std_per_s2"],
                fmt="o",
                color=colors[configuration_name],
                markersize=4,
                capsize=2,
                label=configuration_name,
            )
            plot_axis.hlines(
                mean,
                positions.min() - 0.4,
                positions.max() + 0.4,
                linewidth=2.0,
                color=colors[configuration_name],
            )
        plot_axis.set_title(axis_name + " axis: all approved waveforms")
        plot_axis.set_ylabel("K/I [s$^{-2}$]")
        plot_axis.set_xticks(x)
        plot_axis.set_xticklabels(values["segment_id"], rotation=90, fontsize=7)
        plot_axis.grid(True, alpha=0.25)
        plot_axis.legend(
            ncol=6,
            fontsize=8,
            title="Point = waveform median; error bar = within-waveform standard deviation; line = configuration mean",
            title_fontsize=8,
        )
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_period_amplitude_diagnostics(
    period_rows,
    stability_rows,
    axis_cutoffs,
    common_cutoff,
    output_path,
):
    table = pd.DataFrame(period_rows)
    stability = pd.DataFrame(stability_rows)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), sharex="col")
    configurations = sorted(table["configuration"].unique())
    colors = {
        configuration: plt.get_cmap("tab10")(index)
        for index, configuration in enumerate(configurations)
    }
    for column, axis_name in enumerate(["IN", "OUT"]):
        plot_axis = axes[0, column]
        axis_table = table[table["axis"] == axis_name]
        below_range = axis_table[
            axis_table["within_common_stable_amplitude_range"] == 0
        ]
        plot_axis.scatter(
            below_range["representative_amplitude_deg"],
            below_range[
                "relative_deviation_from_high_amplitude_reference_percent"
            ],
            s=9,
            color="0.75",
            alpha=0.35,
            label="Rejected: below stable amplitude range",
        )
        deviation_outlier = axis_table[
            (axis_table["within_common_stable_amplitude_range"] == 1)
            & (axis_table["within_reference_deviation_tolerance"] == 0)
        ]
        plot_axis.scatter(
            deviation_outlier["representative_amplitude_deg"],
            deviation_outlier[
                "relative_deviation_from_high_amplitude_reference_percent"
            ],
            s=22,
            marker="x",
            color="black",
            linewidths=0.8,
            label="Rejected: outside ±1% reference tolerance",
        )
        for configuration in configurations:
            values = axis_table[
                (axis_table["configuration"] == configuration)
                & (axis_table["used_for_identification"] == 1)
            ]
            plot_axis.scatter(
                values["representative_amplitude_deg"],
                values[
                    "relative_deviation_from_high_amplitude_reference_percent"
                ],
                s=12,
                alpha=0.65,
                color=colors[configuration],
                label="Used: " + configuration,
            )
        plot_axis.axhline(0.0, color="black", linewidth=0.8)
        for sign in [-1.0, 1.0]:
            plot_axis.axhline(
                sign * STABILITY_TOLERANCE_PERCENT,
                color="black",
                linestyle="--",
                linewidth=0.8,
            )
        plot_axis.axvline(
            common_cutoff,
            color="black",
            linestyle="-.",
            linewidth=1.2,
            label=f"Common adopted cutoff = {common_cutoff:.0f} deg",
        )
        plot_axis.set_title(axis_name + ": individual period samples")
        plot_axis.grid(True, alpha=0.25)
        plot_axis.legend(ncol=2, fontsize=7)

        diagnostic_axis = axes[1, column]
        axis_stability = stability[stability["axis"] == axis_name].sort_values(
            "amplitude_bin_lower_deg"
        )
        centers = 0.5 * (
            axis_stability["amplitude_bin_lower_deg"]
            + axis_stability["amplitude_bin_upper_deg"]
        )
        diagnostic_axis.plot(
            centers,
            axis_stability["absolute_deviation_quantile_percent"],
            "o-",
            label="80th percentile of absolute deviation",
        )
        diagnostic_axis.axhline(
            STABILITY_TOLERANCE_PERCENT,
            color="black",
            linestyle="--",
            linewidth=0.8,
            label="Stability criterion = 1%",
        )
        diagnostic_axis.axvline(
            axis_cutoffs[axis_name],
            color="tab:blue",
            linestyle=":",
            linewidth=1.3,
            label=f"{axis_name} criterion cutoff = {axis_cutoffs[axis_name]:.0f} deg",
        )
        diagnostic_axis.axvline(
            common_cutoff,
            color="black",
            linestyle="-.",
            linewidth=1.2,
            label=f"Common adopted cutoff = {common_cutoff:.0f} deg",
        )
        diagnostic_axis.set_xlabel("Representative amplitude [deg]")
        diagnostic_axis.set_title(axis_name + ": amplitude-band stability check")
        diagnostic_axis.grid(True, alpha=0.25)
        diagnostic_axis.legend(fontsize=7)
    axes[0, 0].set_ylabel(
        "Corrected K/I deviation from high-amplitude reference [%]"
    )
    axes[1, 0].set_ylabel("80th-percentile absolute deviation [%]")
    figure.suptitle(
        "Period-amplitude diagnostic and data-driven period selection\n"
        "Positive K/I deviation means a shorter measured period; negative means longer"
    )
    figure.tight_layout()
    save_figure(figure, output_path)


def write_report(
    output_path,
    period_rows,
    waveform_rows,
    configuration_rows,
    base_rows,
    parameter_rows,
    direction_rows,
    stability_rows,
    axis_cutoffs,
    common_cutoff,
):
    periods = pd.DataFrame(period_rows)
    waveforms = pd.DataFrame(waveform_rows)
    configurations = pd.DataFrame(configuration_rows)
    bases = pd.DataFrame(base_rows)
    parameters = pd.DataFrame(parameter_rows)
    directions = pd.DataFrame(direction_rows)
    used_periods = periods[periods["used_for_identification"] == 1]
    largest_period_residual = periods.iloc[
        periods[
            "relative_deviation_from_high_amplitude_reference_percent"
        ].abs().argmax()
    ]
    maximum_direction = directions.iloc[
        directions["relative_direction_difference_percent"].abs().argmax()
    ]
    calibration_only = configurations[configurations["use_for_calibration"] == 1]
    method_labels = {
        "base plus known spacer increments": "基準値と既知スペーサ増分の和",
        "known gravity restoring divided by measured K/I": "既知の重力復元係数を実測K/Iで除算",
    }
    lines = [
        "# Stage 2: 非線形周期によるI、K同定",
        "",
        "## 結論",
        "",
        "Stage 1で確定した46波形の頂点と中心を使用し、同符号頂点間周期から有限振幅補正済み",
        "`K/I` を求めた。5 degは診断候補の下限にだけ使用し、同定範囲は角度で決め打ちしていない。",
        f"大振幅基準からの偏差率で安定域を判定した結果、INは{axis_cutoffs['IN']:.0f} deg以上、",
        f"OUTは{axis_cutoffs['OUT']:.0f} deg以上となった。両軸に共通する{common_cutoff:.0f} deg以上を採用し、",
        f"さらに個別周期の偏差が±{STABILITY_TOLERANCE_PERCENT:.0f}%以内のものだけを同定に使用した。",
        "波形全体へのフィッティングによる `I`、`K` の調整は行っていない。",
        "SP00～SP04は既知の追加慣性・追加復元力を使って軸別の基準 `I0`、`K0` を同時同定し、",
        "BALLは既知の重力復元力を実測 `K/I` で割って実効慣性を求めた。",
        "",
        "## period_amplitude_diagnostics.pngの説明",
        "",
        "この図は角度に対する周期そのものを直接プロットした図ではない。横軸は同符号頂点間周期の",
        "両端振幅から求めた代表振幅、上段縦軸は各周期を有限振幅補正して得た `K/I` の",
        "大振幅基準からの偏差率である。正の偏差は周期が基準より短い側、負の偏差は長い側を表す。",
        "実データの低振幅側は主に正偏差であるため、この図は周期の伸長を直接示してはいない。",
        "摩擦、中心推定、頂点時刻誤差のどれが原因かは、この図だけでは分離できない。",
        "",
        "- 色付き丸: `I`、`K` 同定に採用した周期",
        "- 灰色丸: データから決めた安定振幅域より小さいため除外した周期",
        "- 黒い×: 安定振幅域内だが大振幅基準から±1%を超えたため除外した周期",
        "- 上段の横破線: 個別周期の採用許容幅±1%",
        f"- 上段の縦一点鎖線: 両軸共通の採用下限{common_cutoff:.0f} deg",
        "- 下段: 5 deg幅ごとの絶対偏差の80パーセンタイルと1%判定線",
        "",
        "## 同定式",
        "",
        "同符号頂点間の周期を `T`、両端のポテンシャルエネルギー平均に対応する代表振幅を `A` とする。",
        "",
        "```math",
        "K/I = [4 K_elliptic(sin^2(A/2))/T]^2",
        "```",
        "",
        "```math",
        "(K0 + Delta K_j)/(I0 + Delta I_j) = (K/I)_j",
        "```",
        "",
        "各波形では周期サンプルの中央値を採用し、軸・形態の代表値は波形代表値の算術平均とした。",
        "ここでの中央値は採用範囲内に残る測定ばらつきに対するロバスト化に使用する。",
        "",
        "## 周期サンプルの採用方法",
        "",
        f"1. 振幅{CANDIDATE_MINIMUM_AMPLITUDE_DEG:.0f} deg以上を診断候補として抽出する。",
        f"2. 各波形の振幅上位{REFERENCE_AMPLITUDE_FRACTION * 100:.0f}%（最低{REFERENCE_MINIMUM_PERIODS}周期）の中央値を大振幅基準とする。",
        "3. 各周期の有限振幅補正済み `K/I` と大振幅基準との差を百分率で求める。",
        f"4. {AMPLITUDE_BIN_WIDTH_DEG:.0f} deg幅ごとに絶対偏差の{STABILITY_QUANTILE * 100:.0f}パーセンタイルを求め、",
        f"   {STABILITY_CONSECUTIVE_BINS}区間連続で{STABILITY_TOLERANCE_PERCENT:.0f}%以内となる最初の振幅を軸別下限とする。",
        "5. 両軸で同じ条件とするため軸別下限の大きい方を共通下限とし、個別偏差も±1%以内の周期だけを採用する。",
        "",
        "この判定により、低振幅域の系統偏差と、安定域内に孤立して現れる頂点時刻の外れ値を同時に除外する。",
        "",
        "## 使用データ",
        "",
        f"- 波形数: {len(waveforms)}",
        f"- 診断候補周期数: {len(periods)}",
        f"- I、K同定への採用周期数: {len(used_periods)}",
        f"- 除外周期数: {len(periods) - len(used_periods)}",
        f"- 1波形あたり採用周期数の範囲: {waveforms['periods_used'].min()}～{waveforms['periods_used'].max()}",
        f"- 採用代表振幅範囲: {used_periods['representative_amplitude_deg'].min():.2f}～{used_periods['representative_amplitude_deg'].max():.2f} deg",
        f"- 採用周期の有限振幅補正率範囲: {used_periods['finite_amplitude_correction_ratio'].min():.4f}～{used_periods['finite_amplitude_correction_ratio'].max():.4f}",
        "",
        "## 基準I0、K0",
        "",
        "較正誤差は二乗平均平方根誤差、Rは相関係数、R^2は決定係数を表す。",
        "",
        "| 軸 | I0 [kg m^2] | K0 [N m/rad] | 二乗平均平方根誤差 [s^-2] | 相関係数R | 決定係数R^2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for unused_index, row in bases.iterrows():
        lines.append(
            f"| {row['axis']} | {row['base_inertia_kg_m2']:.9f} | "
            f"{row['base_restoring_n_m_per_rad']:.9f} | "
            f"{row['calibration_rmse_per_s2']:.4f} | "
            f"{row['calibration_r']:.6f} | {row['calibration_r_squared']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 形態別の採用I、K",
            "",
            "| 軸 | 形態 | I [kg m^2] | 幾何I [kg m^2] | K [N m/rad] | 実測K/I [s^-2] | モデルK/I [s^-2] | K/I残差 [s^-2] | 決定方法 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for unused_index, row in parameters.sort_values(
        ["axis", "configuration"]
    ).iterrows():
        lines.append(
            f"| {row['axis']} | {row['configuration']} | "
            f"{row['inertia_kg_m2']:.9f} | {row['geometric_inertia_kg_m2']:.9f} | "
            f"{row['restoring_n_m_per_rad']:.9f} | "
            f"{row['k_over_i_measured_per_s2']:.4f} | "
            f"{row['k_over_i_model_per_s2']:.4f} | "
            f"{row['k_over_i_residual_per_s2']:.4f} | "
            f"{method_labels[row['identification_method']]} |"
        )
    lines.extend(
        [
            "",
            "### BALLの決定方法を選んだ理由",
            "",
            "BALLは一つの形態しかないため、周期だけでは `K/I` の比しか得られず、`I` と `K` を独立には決められない。",
            "重力復元係数 `K` は質量・重心位置・重力加速度から直接計算でき、空気の付加慣性は重力復元力を変えない。",
            "一方、球が押しのける空気の付加慣性は動的な `I` に現れる。このため、物理的に直接決められる `K` を固定し、",
            "実測 `K/I` から `I=K/(K/I)` を求める。幾何学的 `I` を固定して `K` を逆算すると、付加慣性を誤って",
            "重力復元係数へ配分することになるため採用しない。今後、既知水平力による静的K測定で独立検証する。",
            "",
            "## 残差の要約",
            "",
            f"- SP00～SP04の最大絶対K/I残差: {calibration_only['k_over_i_residual_per_s2'].abs().max():.4f} s^-2",
            f"- SP00～SP04の最大絶対相対残差: {calibration_only['k_over_i_relative_residual_percent'].abs().max():.3f}%",
            f"- 開始方向P-N差の最大絶対値: {abs(maximum_direction['relative_direction_difference_percent']):.3f}% "
            f"({maximum_direction['axis']} {maximum_direction['configuration']})",
            f"- 診断候補周期の最大偏差: {abs(largest_period_residual['relative_deviation_from_high_amplitude_reference_percent']):.2f}% "
            f"({largest_period_residual['segment_id']}、代表振幅{largest_period_residual['representative_amplitude_deg']:.2f} deg)",
            "- 安定振幅域外および±1%を超える個別周期は同定から除外したが、全周期と除外理由はCSVへ残している。",
            "",
            "## 図",
            "",
            "![I0、K0周期較正](frequency_calibration.png)",
            "",
            "![全波形のK/I](waveform_frequency_overview.png)",
            "",
            "![有限振幅補正後の振幅依存残差](period_amplitude_diagnostics.png)",
            "",
            "## 減衰と周期の扱い",
            "",
            "採用している運動方程式には `b`、`c`、`tau` が含まれるため、数値積分では減衰による周期変化を表現できる。",
            "ただしStage 2の楕円積分式は `b=c=tau=0` の理想振り子式である。したがって全振幅域を使うと、",
            "減衰による周期変化が `I`、`K` に混入し、後段で減衰係数へ補償的に再配分される可能性がある。",
            "",
            "低振幅で相対的に支配的になるのは主にクーロン摩擦 `tau` である。一方、二乗抗力 `c` は速度が大きい",
            "高振幅側ほど強くなるため、高振幅だけなら減衰影響がゼロになるわけではない。そのためStage 2では",
            "角度を任意に決めず、補正済みK/Iが安定する領域をデータから選んだ。",
            "",
            "Stage 4～5でエネルギー損失・1半周期先頂点から `b`、`c`、`tau` を同定した後、これらを固定して",
            "完全な運動方程式で周期を再計算する。減衰による周期補正が無視できない場合は、波形全体の誤差ではなく",
            "周期残差だけを目的として `I`、`K` を更新し、減衰係数を再確認する。この交互確認を収束まで行う。",
            "これにより `I,K` と `b,c,tau` を同時に自由フィットして相互に誤配分することを避ける。",
            "",
            "## Stage 2のレビュー事項",
            "",
            f"1. 偏差率から得た共通下限{common_cutoff:.0f} deg、個別偏差±1%以内の{len(used_periods)}周期を採用するか。",
            "2. SP00～SP04の既知増分から求めた軸別I0、K0を採用するか。",
            "3. 空気の付加慣性をIへ配分するため、BALLのKは重力モデル、Iは実測K/Iから求めるか。",
            "4. 減衰同定後、減衰係数固定の周期残差だけでI、Kを再確認する反復方針を採用するか。",
            "",
            "## 持ち越し事項",
            "",
            "- 減衰が周期へ与える影響は、減衰係数同定後に完全な運動方程式で定量化する。",
            "- I、Kを更新する場合も周期残差だけを使い、全波形誤差を下げる自由変数にはしない。",
            "",
            "## 出力",
            "",
            "- [全周期サンプル](period_samples.csv)",
            "- [波形別K/I](waveform_frequency.csv)",
            "- [形態別K/Iと残差](configuration_frequency.csv)",
            "- [基準I0、K0](base_parameters.csv)",
            "- [形態別I、K](identified_inertia_restoring.csv)",
            "- [開始方向比較](direction_frequency_comparison.csv)",
            "- [振幅帯別の安定性判定](amplitude_stability.csv)",
            "- [実行条件](frequency_settings.json)",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    arguments = parse_arguments()
    parent_result = arguments.result_root / arguments.date
    output_directory = (
        parent_result / "hybrid_identification" / "02_frequency_identification"
    )
    ensure_directory(output_directory)
    turning, waveforms, manifest, input_paths = read_inputs(parent_result)
    period_rows, waveform_rows = identify_waveform_frequencies(turning, waveforms)
    period_rows, stability_rows, axis_cutoffs, common_cutoff = (
        derive_stable_period_selection(period_rows)
    )
    period_rows, waveform_rows = finalize_waveform_frequencies(
        period_rows, waveform_rows
    )
    configuration_rows = summarize_configurations(waveform_rows, manifest)
    base_rows, checked_rows, parameter_rows = identify_inertia_and_restoring(
        configuration_rows
    )
    direction_rows = compare_directions(waveform_rows)

    save_csv(output_directory / "period_samples.csv", period_rows)
    save_csv(output_directory / "waveform_frequency.csv", waveform_rows)
    save_csv(output_directory / "configuration_frequency.csv", checked_rows)
    save_csv(output_directory / "base_parameters.csv", base_rows)
    save_csv(
        output_directory / "identified_inertia_restoring.csv", parameter_rows
    )
    save_csv(
        output_directory / "direction_frequency_comparison.csv", direction_rows
    )
    save_csv(output_directory / "amplitude_stability.csv", stability_rows)
    plot_frequency_calibration(
        checked_rows, base_rows, output_directory / "frequency_calibration.png"
    )
    plot_waveform_frequency(
        waveform_rows,
        checked_rows,
        output_directory / "waveform_frequency_overview.png",
    )
    plot_period_amplitude_diagnostics(
        period_rows,
        stability_rows,
        axis_cutoffs,
        common_cutoff,
        output_directory / "period_amplitude_diagnostics.png",
    )
    write_report(
        output_directory / "FREQUENCY_REPORT.md",
        period_rows,
        waveform_rows,
        checked_rows,
        base_rows,
        parameter_rows,
        direction_rows,
        stability_rows,
        axis_cutoffs,
        common_cutoff,
    )
    settings = {
        "stage": 2,
        "date": arguments.date,
        "candidate_minimum_amplitude_deg": CANDIDATE_MINIMUM_AMPLITUDE_DEG,
        "period_definition": "time between turning points of the same sign",
        "finite_amplitude_correction": "complete elliptic integral using energy-mean endpoint amplitude",
        "high_amplitude_reference_fraction": REFERENCE_AMPLITUDE_FRACTION,
        "high_amplitude_reference_minimum_periods": REFERENCE_MINIMUM_PERIODS,
        "amplitude_bin_width_deg": AMPLITUDE_BIN_WIDTH_DEG,
        "stability_tolerance_percent": STABILITY_TOLERANCE_PERCENT,
        "stability_absolute_deviation_quantile": STABILITY_QUANTILE,
        "stability_consecutive_bins": STABILITY_CONSECUTIVE_BINS,
        "stability_minimum_bin_samples": STABILITY_MINIMUM_BIN_SAMPLES,
        "axis_stable_amplitude_cutoff_deg": axis_cutoffs,
        "adopted_common_amplitude_cutoff_deg": common_cutoff,
        "individual_sample_selection": "amplitude at or above common cutoff and corrected K/I within tolerance of waveform high-amplitude reference",
        "waveform_aggregation": "median of selected corrected K/I period samples",
        "configuration_aggregation": "arithmetic mean of waveform medians",
        "spacer_parameter_method": "shared I0 and K0 with known delta I and delta K",
        "ball_parameter_method": "known gravity restoring and effective I = K / measured K-over-I",
        "input_sha256": {
            str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path)
            for path in input_paths
        },
    }
    (output_directory / "frequency_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("結果: " + str(output_directory))


if __name__ == "__main__":
    main()
