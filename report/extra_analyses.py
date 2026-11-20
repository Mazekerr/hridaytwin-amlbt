"""Extra analyses quoted in the report: alert-budget sensitivity (Table V) and SHAP ranking.

Usage (from the repo root):  python report/extra_analyses.py
Regenerates the cohort with the default seed (about 2 minutes on a laptop CPU), reproduces the
fused out-of-fold predictions and writes model/extra_analyses.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hridaytwin import config                                   # noqa: E402
from hridaytwin.model import (build_dataset, feature_sets, gbm,  # noqa: E402
                              oof_scores, operating_point, rule_score)
from hridaytwin.synth import simulate                           # noqa: E402

ehr, daily, events, latent, truth, cal = simulate()
df, _ = build_dataset(ehr, daily, events, cal)
cols = [c for c in df.columns if not c.startswith("y") and c != "days_to_event"]
sets = feature_sets(cols)
feats = sets["fused"]
y = df["y"].to_numpy()
folds = list(GroupKFold(n_splits=config.N_FOLDS).split(df, y, df["patient_id"].to_numpy()))
p = oof_scores(df, feats, folds)
rules = rule_score(df)
out = {"fused_auroc": round(float(roc_auc_score(y, p)), 4), "alert_budgets": {}}

for budget in (0.05, 0.10, 0.20):
    out["alert_budgets"][f"{int(budget * 100)}_per_100_patient_weeks"] = {
        "fused": operating_point(df, p, budget),
        "rules": operating_point(df, rules + 1e-6 * p, budget)}

# SHAP: production model fitted on all rows, 4,000 randomly sampled check-ins
model = gbm().fit(df[feats], y)
sample = df.sample(4000, random_state=0)
sv = shap.TreeExplainer(model).shap_values(sample[feats])
sv = sv[1] if isinstance(sv, list) else sv
imp = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False)
ehr_f = set(sets["ehr_only"])
cal_f = {"fast_days_next14", "feast_days_next14", "monsoon"}
tot = imp.sum()
share_ehr = imp[[f for f in feats if f in ehr_f]].sum() / tot
share_cal = imp[[f for f in feats if f in cal_f]].sum() / tot
out["shap_top12"] = {k: round(float(v), 4) for k, v in imp.head(12).items()}
out["shap_group_share"] = {"clinic_record": round(float(share_ehr), 3),
                           "calendar": round(float(share_cal), 3),
                           "home_device_stream": round(float(1 - share_ehr - share_cal), 3)}
(ROOT / "model" / "extra_analyses.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
