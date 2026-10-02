# Observer and model-wind follow-up tasks

This roadmap follows completion of the HBK identification work. The first task
records and makes the adopted mechanical coefficients selectable for the next
simulation stage. Tasks OW-02 through OW-05 remain future work.

| ID | Task | Status |
|---|---|---|
| OW-01 | Register the adopted `I`, `K`, `c_ball`, and axis-specific `tau` by axis and configuration; set `b=0`; record source and provisional status; make the values selectable by simulation code. | Complete: registry, selector, and nonlinear plant integrator added; all 12 axis/configuration combinations checked. |
| OW-02 | Rerun model-wind plant/observer comparisons using the adopted coefficients. Compare ESO, RTS, static conversion, and LPF under matched conditions. Mark RTS as an offline method that uses future measurements. | Pending |
| OW-03 | Separate estimator tuning and evaluation wind inputs. Test mean wind, turbulence, and gusts; use the prior 6 m/s maximum-wind case and target TI 20%, and report the realized TI and spectrum. | Pending |
| OW-04 | Evaluate more than RMSE: bias, 95th-percentile and maximum error, phase/delay, gust-peak tracking, and false estimates with no wind. Report direct force/torque error separately from converted wind-speed error. | Pending |
| OW-05 | State the scope of model-wind validation. For real-wind validity, use a calibrated anemometer and account for reference-sensor separation and sample-rate differences. | Pending |

## OW-01 result and use

The adopted values are stored in `config/hbk_model_coefficients.json`. Select
the motion axis and setup explicitly; for example:

```bash
python 06_Analysis/simulation/src/hbk_model_coefficients.py --axis IN --configuration BALL
```

The selector returns inertia, restoring stiffness, rod and sphere quadratic
drag, viscous damping, and the one-integral Coulomb-friction coefficient. It
rejects missing cases and nonzero viscous damping. `simulate_hbk_plant()` in
the same module integrates the adopted nonlinear plant with RK4. It takes
wind-generated force as input and includes quadratic drag and smoothed
Coulomb friction. Supply the force lever explicitly; it is not inferred from
the mechanical inertia registry. The registry includes the
five spacer configurations (`SP00` to `SP04`) and `BALL`, for both `IN` and
`OUT` axes.

The I/K values for BALL use the accepted 180.61 mm effective ball-center arm
and fixed 3.9 g sphere mass. The ball arm was selected from the free-decay
period fit; it remains an effective model value until the physical center
distance is independently measured. The sphere drag was identified with
Method A and has an equivalent drag coefficient near 0.486. Rod drag is the
theoretical value, corrected for exposed rod length in the BALL configuration.
The friction coefficient is axis-specific. All records use `b=0`, consistent
with the adopted identification model.

The older `config/real_wind_doe.json` and its result folder are historical
Stage 2 outputs based on a prior single-configuration model. They are retained
as provenance and are not silently relabeled as runs with adopted HBK
coefficients. The new registry, selector, and plant integrator are the input
for OW-02, which will connect these coefficients to matched wind/observer
comparisons. OW-01 establishes the plant-side implementation; it does not
claim updated observer comparison results.

## Coefficient provenance

| Quantity | Source and interpretation |
|---|---|
| `I`, `K` for SP00–SP04 | IHB-02 free-decay frequency identification by axis and setup. |
| `I`, `K` for BALL | IHB-05 position-adjusted values for 180.61 mm effective arm, 3.9 g sphere. |
| `c_rod` | IHB-04 theoretical cylinder-drag estimate. |
| `c_ball` | IHB-05 Method A monotone-amplitude fit, common across axes. |
| `tau` | IHB-03 monotone-smoothed amplitude, one-integral fit, one value per axis; reused by IHB-05. |
| `b` | Fixed at zero by the adopted identification assumptions. |

## OW-01 checks

The registry contains 12 unique cases. A command-line selection test is run
for all 12; BALL returns adjusted inertia and stiffness, while spacer cases
preserve IHB-02 values. The plant integrator is smoke-tested with a zero-force,
nonzero-angle free response and produces a finite trajectory. The old Stage 2
DOE is not rerun here; matched observer comparisons belong to OW-02.
