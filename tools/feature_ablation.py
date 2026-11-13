"""Which optional ideas help? Runs TableCast's own training + 7-day backtest with:
  off        every option off (the current model)
  all        every option on
  all-<x>    every option on except <x>   (leave-one-out: if removing x improves things, x hurts)
  only-<x>   only option x on            (with --single)

Datasets are CSV:City:COUNTRY (country may include a state, e.g. DE-BW):
    py tools/feature_ablation.py data.csv:Stuttgart:DE-BW kiel.csv:Kiel:DE-SH --model recommended
    py tools/feature_ablation.py ... --holdout   # score an EARLIER period (drops the last 25% of history)

Safeguard: an option is only flagged for removal if dropping it lowers the average 7-day error by
>= MIN_GAIN points AND no dataset gets worse by more than MAX_LOSS points.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import core  # noqa: E402

MIN_GAIN = 0.2
MAX_LOSS = 0.3


def load(path, holdout=False):
    visits, _, error = core.clean_visits(core.read_csv_bytes(Path(path).read_bytes()))
    if error:
        raise SystemExit(f"{path}: {error[0]}")
    if holdout:  # drop the last 25%: the backtest then scores an earlier period
        hist = core.history_only(visits)
        cutoff = hist["visit_date"].iloc[int(len(hist) * 0.75)]
        visits = visits[visits["visit_date"] < cutoff].reset_index(drop=True)
    return visits


def configs(single):
    opts = list(core.FEATURE_OPTIONS)
    out = {"off": dict(core.NO_OPTIONS), "all": {k: True for k in opts}}
    for k in opts:
        out[f"all-{k}"] = {o: o != k for o in opts}
    if single:
        for k in opts:
            out[f"only-{k}"] = {o: o == k for o in opts}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--model", default="recommended")
    parser.add_argument("--single", action="store_true", help="also run each option on its own")
    parser.add_argument("--holdout", action="store_true", help="score an earlier period instead")
    parser.add_argument("--out", help="write all results as JSON here")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    rows = []
    for spec in args.datasets:
        path, city, country = spec.rsplit(":", 2)
        visits = load(path, args.holdout)
        for name, options in configs(args.single).items():
            t0 = time.perf_counter()
            b = core.train_bundle(visits, city, country, model_spec=args.model, options=options)
            rows.append({"dataset": Path(path).stem, "config": name, "wape7": b["wape7"], "wape1": b["wape"],
                         "seconds": time.perf_counter() - t0})
            print(f"{Path(path).stem:34} {name:20} 7-day {b['wape7']:6.2f}%  next-day {b['wape']:6.2f}%  "
                  f"{rows[-1]['seconds']:5.1f}s", flush=True)

    df = pd.DataFrame(rows)
    off = df[df.config == "off"].set_index("dataset")["wape7"]
    df["vs_off"] = df.apply(lambda r: r.wape7 - off[r.dataset], axis=1)
    summary = df.groupby("config").agg(mean_7d=("wape7", "mean"), mean_change_vs_off=("vs_off", "mean"),
                                       better_than_off=("vs_off", lambda s: int((s < -0.05).sum())),
                                       worse_than_off=("vs_off", lambda s: int((s > 0.05).sum())),
                                       worst_change=("vs_off", "max"), seconds=("seconds", "mean"))
    print(f"\nSUMMARY over {df.dataset.nunique()} datasets (7-day error; negative change = better than 'off')")
    print(summary.sort_values("mean_7d").round(2).to_string())
    full = df[df.config == "all"].set_index("dataset")["wape7"]
    print("\nLEAVE-ONE-OUT: removing the option changes the 7-day error by (negative = the option HURTS)")
    for k in core.FEATURE_OPTIONS:
        change = df[df.config == f"all-{k}"].set_index("dataset")["wape7"] - full
        if change.empty:
            continue
        verdict = "REMOVE?" if change.mean() <= -MIN_GAIN and change.max() <= MAX_LOSS else "keep"
        print(f"  {k:14} mean {change.mean():+.2f} pts  worst dataset {change.max():+.2f}  -> {verdict}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
