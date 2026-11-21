"""
Plot per-block RMSE for one stream + seed, and optionally compare against reference
results (e.g. the teammate's CSVs in results/) to validate the rebuild.

    python plot_check.py --results OUT --stream sudden --seed 0 --reference results --out check.png

Solid lines = --results. Dashed lines = --reference. Grey dotted = true drift start.
Triangles = blocks where a policy raised drift_flag.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from data.streams import drift_blocks_for
from methods.policies import POLICIES


def read(folder, stream, seed, method):
    f = Path(folder) / f"{stream}_{seed}_{method}.csv"
    return pd.read_csv(f).set_index("block") if f.exists() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--stream", default="sudden")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--methods", nargs="+", default=list(POLICIES))
    p.add_argument("--reference", default=None, help="folder with CSVs to compare against")
    p.add_argument("--out", default="check.png")
    a = p.parse_args()

    fig, ax = plt.subplots(figsize=(10, 5))
    rows = []
    for i, m in enumerate(a.methods):
        new = read(a.results, a.stream, a.seed, m)
        if new is None:
            continue
        color = f"C{i}"
        ax.plot(new.index, new.rmse, color=color, label=m)
        fl = new[new.drift_flag == 1]
        ax.scatter(fl.index, fl.rmse, marker="v", s=80, color=color)
        row = {"method": m, "pre(25-29)": new.rmse.loc[25:29].mean(),
               "block30": new.rmse.get(30), "post(31-59)": new.rmse.loc[31:].mean()}
        if a.reference:
            ref = read(a.reference, a.stream, a.seed, m)
            if ref is not None:
                ax.plot(ref.index, ref.rmse, color=color, ls="--", alpha=0.6, label=f"{m} (reference)")
                diff = (new.rmse - ref.rmse).abs()
                row.update({"ref_post(31-59)": ref.rmse.loc[31:].mean(),
                            "mean|diff|": diff.mean(), "max|diff|": diff.max(),
                            "max|diff| block": int(diff.idxmax())})
        rows.append(row)

    if not rows:
        raise SystemExit(f"no CSVs for {a.stream} seed {a.seed} in {a.results}")
    for d in drift_blocks_for(a.stream):
        ax.axvline(d, color="grey", ls=":")
    ax.set(xlabel="block", ylabel="RMSE", title=f"{a.stream} stream, seed {a.seed}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("saved", a.out)


if __name__ == "__main__":
    main()
