"""Build the figures used in the AMLBT report (report/figures/*.pdf).

Reads only pipeline outputs, so every number in a figure comes from the code:
    model/metrics.json, model/twin_validation.json, dashboard/data.json

Usage (from the repository root, after `python run_pipeline.py`):
    python report/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREY, RED = "#2a78d6", "#eb6834", "#9a9893", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL_W = 3.45  # IEEE single-column width in inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.edgecolor": GREY,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2)


def performance(m):
    v = m["variants"]
    names = [("Threshold rules", v["clinical_rules"]["auroc"], GREY),
             ("Wearable + home only", v["stream_only"]["auroc"], GREY),
             ("EHR only", v["ehr_only"]["auroc"], ORANGE),
             ("Fused + twin state", v["fused_plus_twin_state"]["auroc"], GREY),
             ("Fused, logistic", v["fused_logistic"]["auroc"], GREY),
             ("Fused GBM (ours)", v["fused"]["auroc"], BLUE)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W * 2.05, 1.95), gridspec_kw={"width_ratios": [1, 1.15]})
    y = np.arange(len(names))
    a.barh(y, [n[1] - 0.5 for n in names], left=0.5, color=[n[2] for n in names], height=0.62)
    for i, (lab, val, _) in enumerate(names):
        a.text(val + 0.004, i, f"{val:.3f}", va="center", fontsize=6.8, color=INK)
    a.set_yticks(y, [n[0] for n in names])
    a.set_xlim(0.5, 0.78)
    a.set_xlabel("AUROC, 14-day composite event (5-fold, grouped by patient)")
    a.set_title("(a) Model comparison", loc="left", color=INK)
    style(a, "x")

    et = m["event_types"]
    keys = ["severe_hypoglycaemia", "hyperglycaemic_crisis", "hypertensive_emergency", "acute_cv_event"]
    labels = ["Severe\nhypo.", "Hyper-\nglycaemic", "Hyper-\ntensive", "Acute\nCV"]
    x = np.arange(len(keys))
    w = 0.36
    ehr = [et[k]["ehr_only_auroc"] for k in keys]
    fus = [et[k]["fused_auroc"] for k in keys]
    b.bar(x - w / 2 - 0.01, np.array(ehr) - 0.5, w, bottom=0.5, color=ORANGE, label="EHR only")
    b.bar(x + w / 2 + 0.01, np.array(fus) - 0.5, w, bottom=0.5, color=BLUE, label="Fused (ours)")
    for i, k in enumerate(keys):
        p = et[k]["delta_vs_ehr"]["p_value"]
        d = et[k]["delta_vs_ehr"]["delta"]
        b.text(i, max(ehr[i], fus[i]) + 0.012, f"{d:+.3f}\np={p:.3f}", ha="center", fontsize=6, color=INK2)
    b.set_xticks(x, labels)
    b.set_ylim(0.5, 1.08)
    b.set_ylabel("AUROC")
    b.legend(frameon=False, loc="lower right", ncol=2, bbox_to_anchor=(1.0, 0.99), fontsize=6.3, borderaxespad=0)
    b.set_title("(b) By event type", loc="left", color=INK)
    style(b)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(OUT / "fig_performance.pdf")
    plt.close(fig)


def twin(tv, m):
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.7))
    for key, col, lab in (("glucose", BLUE, "Glucose"), ("blood_pressure", ORANGE, "Systolic BP")):
        wk = [0] + [c["week"] for c in tv[key]["checkpoints"]]
        r = [tv[key]["prior_corr_with_truth"]] + [c["corr_with_truth"] for c in tv[key]["checkpoints"]]
        a.plot(wk, r, color=col, lw=1.6, marker="o", ms=3, label=lab)
    a.set_xlabel("Weeks of home data")
    a.set_ylabel("Corr. with true response")
    a.set_ylim(-0.03, 0.75)
    a.legend(frameon=False, loc="lower right")
    a.set_title("(a) Twin learns the patient", loc="left", color=INK)
    style(a)

    rel = m["reliability"]
    xp = [r["mean_pred"] for r in rel]
    yo = [r["observed"] for r in rel]
    lim = max(max(xp), max(yo)) * 1.08
    b.plot([0, lim], [0, lim], color=GREY, lw=0.8, ls="--")
    b.plot(xp, yo, color=BLUE, lw=1.6, marker="o", ms=3)
    b.set_xlim(0, lim)
    b.set_ylim(0, lim)
    b.set_xlabel("Predicted 14-day risk")
    b.set_ylabel("Observed rate")
    b.set_title("(b) Risk calibration", loc="left", color=INK)
    style(b, "both")
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(OUT / "fig_twin_calibration.pdf")
    plt.close(fig)


def case_study(dash, pid="HT00000"):
    p = next(q for q in dash["patients"] if q["id"] == pid)
    s = p["series"]
    days = np.arange(len(s["sbp"]))
    sbp = np.array([np.nan if v is None else v for v in s["sbp"]], float)
    mu = np.array([np.nan if v is None else v for v in s["twin_b"]], float)
    sd = np.array([np.nan if v is None else v for v in s["twin_b_sd"]], float)
    rd = np.array(p["risk"]["days"])
    rp = np.array([np.nan if v is None else v for v in p["risk"]["p"]], float)
    ev = [e["day"] for e in p["events"]]
    thr = dash["threshold"]

    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 2.45), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]})
    a.fill_between(days, mu - 2 * sd, mu + 2 * sd, color=BLUE, alpha=0.18, lw=0, label="Twin expectation (±2 sd)")
    out = np.abs(sbp - mu) > 2 * sd
    a.scatter(days[~out], sbp[~out], s=5, color=INK2, lw=0, label="Home SBP reading")
    if out.any():
        a.scatter(days[out], sbp[out], s=9, color=ORANGE, lw=0, label="Outside band")
    a.set_ylabel("SBP (mmHg)")
    a.legend(frameon=False, loc="upper left", ncol=2, fontsize=6, handletextpad=0.2, columnspacing=0.8,
             bbox_to_anchor=(0, 1.22))
    style(a)
    b.plot(rd, rp * 100, color=BLUE, lw=1.4)
    b.axhline(thr * 100, color=GREY, ls="--", lw=0.8)
    b.text(1, thr * 100 - 0.6, f"alert line\n{thr * 100:.1f}%", fontsize=6, color=INK2, ha="left", va="top")
    b.set_ylabel("14-day risk (%)")
    b.set_xlabel("Day of follow-up")
    style(b)
    for ax in (a, b):
        for e in ev:
            ax.axvline(e, color=RED, lw=1.1)
    b.text(ev[0] + 2, np.nanmax(rp) * 100 * 0.92, "emergency\nadmission", fontsize=6, color=RED, va="top")
    b.set_xlim(0, len(days))
    fig.tight_layout(pad=0.3, h_pad=0.3)
    fig.savefig(OUT / "fig_case_study.pdf")
    plt.close(fig)


def main():
    m = json.loads((ROOT / "model" / "metrics.json").read_text())
    tv = json.loads((ROOT / "model" / "twin_validation.json").read_text())
    dash = json.loads((ROOT / "dashboard" / "data.json").read_text())
    performance(m)
    twin(tv, m)
    case_study(dash)
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
