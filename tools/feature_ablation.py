"""Which optional ideas help? Runs TableCast's own training + 7-day backtest with:
  off        every option off (the current model)
  all        every option on
  all-<x>    every option on except <x>   (leave-one-out: if removing x improves things, x hurts)
  only-<x>   only option x on            (with --single)

Datasets are CSV:City:COUNTRY (country may include a state, e.g. DE-BW):
    py tools/feature_ablation.py data.csv:Stuttgart:DE-BW kiel.csv:Kiel:DE-SH --model recommended
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


def load(path):
    visits, _, error = core.clean_visits(core.read_csv_bytes(Path(path).read_bytes()))
    if error:
        raise SystemExit(f"{path}: {error[0]}")
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
    parser.add_argument("--out", help="write all results as JSON here")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    rows = []
    for spec in args.datasets:
        path, city, country = spec.rsplit(":", 2)
        visits = load(path)
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
    all_mean = summary.loc["all", "mean_7d"]
    print("\nLEAVE-ONE-OUT: removing the option changes the average 7-day error by (negative = the option HURTS)")
    for k in core.FEATURE_OPTIONS:
        if f"all-{k}" in summary.index:
            print(f"  {k:14} {summary.loc[f'all-{k}', 'mean_7d'] - all_mean:+.2f} pts")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
