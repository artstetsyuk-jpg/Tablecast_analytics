"""Compare the forecasting model on this branch with the one on another git branch, on real data.

Both models run through TableCast's own training and 7-day rolling backtest (core.train_bundle), so
the comparison is fair: same features, same data split, only the model differs.

Each dataset is a CSV with daily guest counts (any column names the app understands, e.g.
visit_date/visitors or date/covers), plus the city and country used for weather and holidays:

    py tools/compare_models.py --against test  data/locations/frenchbakery/visits.csv:Paris:FR  cafe_b.csv:Vienna:AT

Prints the 7-day and next-day error for the old and the new model per dataset, and a summary.
"""
import argparse
import importlib.util
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import core  # noqa: E402  (this branch's model)


def load_core_from_branch(ref):
    source = subprocess.run(["git", "show", f"{ref}:core.py"], cwd=ROOT, check=True,
                            capture_output=True, text=True, encoding="utf-8").stdout
    path = Path(tempfile.mkdtemp()) / "core_other.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("core_other", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_visits(path):
    raw = core.read_csv_bytes(Path(path).read_bytes())
    visits, _, error = core.clean_visits(raw)
    if error:
        raise SystemExit(f"{path}: cannot use this file ({error[0]})")
    return visits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--against", default="test", help="git branch/commit with the model to compare against")
    parser.add_argument("datasets", nargs="+", help="CSV:City:COUNTRY, e.g. visits.csv:Vienna:AT")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    other = load_core_from_branch(args.against)

    rows = []
    for spec in args.datasets:
        path, city, country = spec.rsplit(":", 2)
        visits = load_visits(path)
        old = other.train_bundle(visits, city, country)
        new = core.train_bundle(visits, city, country)
        rows.append({"dataset": Path(path).name, "old_7d": old["wape7"], "new_7d": new["wape7"],
                     "old_1d": old["wape"], "new_1d": new["wape"], "baseline_7d": new["base_wape7"]})
        r = rows[-1]
        print(f"{r['dataset']:28} 7-day {r['old_7d']:5.1f}% -> {r['new_7d']:5.1f}%   "
              f"next-day {r['old_1d']:5.1f}% -> {r['new_1d']:5.1f}%   (last-week guess {r['baseline_7d']:5.1f}%)")

    df = pd.DataFrame(rows)
    better = int((df.new_7d < df.old_7d).sum())
    print(f"\n7-day error on average: {args.against} {df.old_7d.mean():.2f}% -> this branch {df.new_7d.mean():.2f}% "
          f"(better on {better} of {len(df)})")


if __name__ == "__main__":
    main()
