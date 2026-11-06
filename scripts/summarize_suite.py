"""One compact markdown summary of an offline-suite output folder (results/SUMMARY.md)."""
import glob, sys
from pathlib import Path
import pandas as pd

od = Path(sys.argv[1])
out = []
for exp, title in (("B", "Experiment B: image shifts (same test episodes in every column)"),
                   ("A", "Experiment A: policy shift (calibrated on teleop)")):
    fs = sorted(glob.glob(str(od / f"table_{exp}_*_*.csv")))
    if not fs:
        continue
    T = pd.concat([pd.read_csv(f) for f in fs])
    out.append(f"## {title}\n")
    n = T.groupby("condition")[["n_eval", "n_eval_success", "n_eval_fail"]].first()
    out.append("Evaluation episodes per condition (after the first k):\n\n" + n.to_markdown() + "\n")
    for val in ("false_alarm", "detection"):
        pv = T[T.method.isin(["fixed", "self_ref", "recal_k", "adaptive_log"])].pivot_table(
            index=["encoder", "variant", "method"], columns="condition", values=val)
        out.append(f"### {val.replace('_', ' ')} rate\n\n" + pv.round(3).to_markdown() + "\n")
    au = T[T.method == "fixed"].pivot_table(index=["encoder", "variant"], columns="condition", values="auroc_raw")
    out.append("### AUROC (failure vs success ranking)\n\n" + au.round(3).to_markdown() + "\n")
(od / "SUMMARY.md").write_text("\n".join(out))
print("wrote", od / "SUMMARY.md")
