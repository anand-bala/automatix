"""Run all experiments for HSCC 2025."""

import argparse
import itertools
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence, TypeAlias

import networkx as nx

from automatix.afa.strel import StrelAutomaton

N_TRIALS = 7
N_REPEATS = 10


CURRENT_DIR = Path(__file__).parent

Location: TypeAlias = int
Alph: TypeAlias = "nx.Graph[Location]"


def import_from_path(module_name: str, file_path: Path) -> ModuleType:
    import importlib.util

    assert file_path.is_file()

    module_spec = importlib.util.spec_from_file_location(module_name, file_path.absolute())
    assert module_spec is not None
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    return module


monitoring_utils = import_from_path("monitoring_utils", CURRENT_DIR / "monitoring_example.py")


EXPERIMENTS = [
    # Map, Trace
    (  # Map 1
        "./data/Map_1/map_params.json",
        "./data/Map_1/log_2024_11_12_14_09_20.csv",
    ),
    (  # Map 2
        "./data/Map_2/map_params.json",
        "./data/Map_2/log_2024_11_12_13_53_21.csv",
    ),
    (  # Map 3
        "./data/Map_3/map_params.json",
        "./data/Map_3/log_2024_11_12_17_52_42.csv",
    ),
    (  # Map 4
        "./data/Map_4/map_params.json",
        "./data/Map_4/log_2024_11_12_16_54_58.csv",
    ),
    (  # Map 5
        "./data/Map_5/map_params.json",
        "./data/Map_5/log_2024_11_12_17_10_39.csv",
    ),
]

SPECS = [
    "./establish_comms_spec.py",
    "./reach_avoid_spec.py",
]


def forward_run(
    monitor: StrelAutomaton,
    trace: Sequence["nx.Graph[Location]"],
    ego_locs: Mapping[str, int],
) -> dict[str, bool]:
    states = {name: monitor.initial_at(loc) for name, loc in ego_locs.items()}
    for input in trace:
        states = {name: monitor.next(input, state) for name, state in states.items()}
    return {name: monitor.final_weight(state) for name, state in states.items()}


@dataclass
class Args:
    @classmethod
    def parse_args(cls) -> "Args":
        parser = argparse.ArgumentParser(description="Run all experiments for HSCC 2025")

        args = parser.parse_args()
        return Args(**vars(args))


def main(args: Args) -> None:
    for spec, (map_file, trace_file), online in itertools.product(
        map(lambda p: Path(CURRENT_DIR, p), SPECS),
        map(lambda p: (Path(CURRENT_DIR, p[0]), Path(CURRENT_DIR, p[1])), EXPERIMENTS),
        [False, True],
    ):
        experiment_script = [
            sys.executable,
            Path(CURRENT_DIR, "./monitoring_example.py"),
            "--timeit",
            "--spec",
            spec,
            "--map",
            map_file,
            "--trace",
            trace_file,
        ]
        if online:
            experiment_script.append("--online")
        subprocess.run(experiment_script)


if __name__ == "__main__":
    main(Args.parse_args())
