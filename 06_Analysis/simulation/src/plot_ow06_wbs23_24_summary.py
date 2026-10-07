"""Plot compact time-series summaries for the WBS 2.3 and 2.4 reports."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
SIM=HERE.parent
DESIGN=SIM/"results/observer_wind/ow06_frequency_aware_observer_design"

INPUTS=(
    ("1Hz_sine","1 Hz sinusoidal wind"),
    ("独立Kaimal乱流 平均2 m/s TI20%","Kaimal turbulence"),
    ("ガスト 2→6 m/s","Gust, 2 to 6 m/s"),
    ("自由振動_30deg_release","Free decay, 30 deg release"),
)

WBS23_METHODS=(
    ("RTS_fixed_q1","Fixed-Q RTS","#3b4a6b"),
    ("RTS_frequency_from_record","Frequency-selected Q","#2b8cbe"),
    ("RTS_estimated_frequency_harmonic_state","Tone-state RTS","#1b9e77"),
    ("RTS_calibrated_coeff_q1","Calibrated coefficients","#d95f02"),
)
WBS24_METHODS=(
    ("raw","Fixed observer, raw","#404040"),
    ("boxcar_0.25s_zero_phase","Boxcar 0.25 s","#2b8cbe"),
    ("boxcar_0.5s_zero_phase","Boxcar 0.5 s","#1b9e77"),
    ("notch_Q3","Notch Q3","#d73027"),
    ("notch_Q5","Notch Q5","#756bb1"),
    ("notch_Q10","Notch Q10","#e08214"),
)


def x_window(input_name,t):
    if input_name=="1Hz_sine": return (20.,25.)
    if "Kaimal" in input_name: return (15.,30.)
    if "ガスト" in input_name: return (float(t[0]),float(t[-1]))
    return (.5,min(12.,float(t[-1])))


def make_figure(folder,version,method_defs,axis,filename):
    bundle=np.load(folder/"timeseries.npz",allow_pickle=False)
    records=json.loads(str(bundle["__metadata__"]))
    fig,axs=plt.subplots(4,4,figsize=(19,11),sharex=False,layout="constrained")
    panels=(
        ("matched","ideal","Matched | ideal sensor"),
        ("matched","OW05_noise_seed0","Matched | OW-05 noise"),
        ("OW04_plant_mismatch","ideal","OW-04 mismatch | ideal sensor"),
        ("OW04_plant_mismatch","OW05_noise_seed0","OW-04 mismatch | OW-05 noise"),
    )
    for row,(input_name,input_label) in enumerate(INPUTS):
        for col,(plant,sensor,panel_label) in enumerate(panels):
            ax=axs[row,col]
            case=[m for m in records if m["axis"]==axis and m["input"]==input_name
                  and m["plant_case"]==plant and m["sensor_case"]==sensor]
            if not case:
                ax.set_visible(False); continue
            first=case[0]; t=bundle[f"{first['case_id']}_t"]; truth=bundle[f"{first['case_id']}_truth"]
            ax.plot(t,truth,color="black",lw=2.,label="True wind",zorder=10)
            for method,label,color in method_defs:
                rec=next((m for m in case if m["method"]==method),None)
                if rec is not None:
                    ax.plot(t,bundle[rec["estimate_id"]],color=color,lw=1.,alpha=.95,label=label)
            lo,hi=x_window(input_name,t); ax.set_xlim(lo,hi)
            ax.grid(True,alpha=.24); ax.set_ylabel("Wind [m/s]")
            ax.set_title(f"{input_label} | {panel_label}",loc="left",fontsize=8)
            if row==3: ax.set_xlabel("Time [s]")
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,fontsize=8,frameon=False)
    fig.suptitle(f"{version} time-series summary | {axis} axis | all four input cases and plant/sensor combinations\n"
                 "Original time axis; no lag or amplitude correction applied to curves",fontsize=14)
    out=folder/filename
    fig.savefig(out,dpi=120,bbox_inches="tight")
    plt.close(fig); bundle.close()
    return out


def main():
    paths=[]
    for axis in ("IN","OUT"):
        paths.append(make_figure(DESIGN/"wbs23_offline","WBS 2.3",WBS23_METHODS,axis,f"timeseries_summary_{axis}.png"))
        paths.append(make_figure(DESIGN/"wbs24_output_postprocess","WBS 2.4",WBS24_METHODS,axis,f"timeseries_summary_{axis}.png"))
    for p in paths: print(p)


if __name__=="__main__": main()
