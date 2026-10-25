"""Parts for the mistake DB, drawn from data outside every test set.

    python -m experiments.rag.samples N_ECCV N_ITERCAD

Prints `<input config> <ids, comma separated>` per pool, for run_parallel.sh
--datasets/--samples. A pool is shuffled once with a fixed seed, so a larger N
keeps the parts of a smaller one.
"""

import csv
import random
import sys
from pathlib import Path

DATA = Path("data")
SEED = 0


def test_ids(data: Path = DATA) -> set[str]:
    """Every part a test set evaluates; none may reach the DB."""
    stems = {
        path.name.split(".")[0].removesuffix("_dim")
        for folder in ("test_eccv/techdraw", "test_vlm/techdraw", "test_itercad/techdraw",
                       "test_ortho2cad_f360_mdim/techdraw")
        for path in (data / folder).iterdir()
    }
    assert stems, f"no test sets under {data}"
    return stems


def eccv_pool(data: Path = DATA) -> list[str]:
    train = data / "eccv2026-cad-challenge-data/train"
    drawn = {path.name.removesuffix("_dim.png") for path in (train / "techdraw_dimensioned").glob("*_dim.png")}
    return sorted({path.stem for path in (train / "target_step_ori").glob("*.step")} & drawn)


def itercad_pool(data: Path = DATA) -> list[str]:
    """Hard parts only: the test set is 49 hard and 1 easy."""
    with open(data / "test_itercad_ori/dataset_index.csv") as index:
        return sorted(row["sample_id"] for row in csv.DictReader(index) if row["level"] == "hard")


def draw(pool: list[str], n: int, excluded: set[str]) -> list[str]:
    allowed = [part for part in pool if part not in excluded]
    random.Random(SEED).shuffle(allowed)
    return allowed[:n]


if __name__ == "__main__":
    excluded = test_ids()
    for config, pool, n in (("train_eccv", eccv_pool(), sys.argv[1]), ("itercad_pool", itercad_pool(), sys.argv[2])):
        print(config, ",".join(draw(pool, int(n), excluded)))
