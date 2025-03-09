import argparse
import cProfile
import csv
import itertools
import json
import math
import timeit
from collections import deque
from pathlib import Path
from time import perf_counter_ns
from typing import Mapping, MutableSequence, Self, Sequence, TypeAlias

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field, model_validator

from automatix.afa.strel import StrelAutomaton, make_bool_automaton
from automatix.logic import strel

DRONE_COMMS_RADIUS: float = 40
GCS_COMMS_RADIUS: float = 60
CLOSE_TO_GOAL_THRESH = 0.5
CLOSE_TO_OBSTACLE_THRESH = 0.1

N_TRIALS = 10
N_REPEATS = 7

Location: TypeAlias = int
Alph: TypeAlias = "nx.Graph[Location]"


class GroundStation(BaseModel):
    id: int
    y: float = Field(alias="north")
    x: float = Field(alias="east")


class Building(BaseModel):
    id: int
    y: float = Field(alias="north")
    x: float = Field(alias="east")


class GoalPosition(BaseModel):
    x: float
    y: float


class MapInfo(BaseModel):
    num_blocks: int = Field(alias="nb_blocks")
    street_width_perc: float
    building_width: float
    street_width: float
    radius_of_influence: float = GCS_COMMS_RADIUS


class Map(BaseModel):
    map_properties: MapInfo
    buildings: list[Building]
    ground_stations: list[GroundStation]
    goal_position: GoalPosition = Field(alias="goal_positions")


class TraceSample(BaseModel):
    drone: int = Field(alias="Drone")
    time: float = Field(alias="Time")
    pos_y: float = Field(alias="North")
    pos_x: float = Field(alias="East")
    pos_z: float = Field(alias="Altitude")
    vel_x: float = Field(alias="Vx")
    vel_y: float = Field(alias="Vy")


class Args(BaseModel):
    spec_file: Path = Field(description="Path to a file with a STREL specification (python file)", alias="spec")
    map_info: Path = Field(description="Path to map.json file", alias="map")
    trace: Path = Field(description="Path to trace.csv file")

    ego_loc: list[str] = Field(
        description="Names of the ego location. Format is '(drone|groundstation)_(0i)', where `i` is the index. Default: all locations",
        default_factory=list,
        alias="ego",
    )
    timeit: bool = Field(description="Record performance", default=False)
    num_trials: int = Field(description="Number of timing trials to run.", default=N_TRIALS)
    num_repeats: int = Field(description="Number of repeats per trial", default=N_REPEATS)
    online: bool = Field(description="Report online monitoring performance", default=False)
    profile: bool = Field(description="Profile the monitoring code", default=False)

    @classmethod
    def parse(cls) -> "Args":
        parser = argparse.ArgumentParser(
            description="Run boolean monitoring example",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--spec", help="Path to a file with a STREL specification (python file)", type=lambda arg: Path(arg), required=True
        )
        parser.add_argument("--map", help="Path to map.json file", type=lambda arg: Path(arg), required=True)
        parser.add_argument("--trace", help="Path to trace.csv file", type=lambda arg: Path(arg), required=True)
        parser.add_argument(
            "--ego",
            help="Name of the ego location. Format is '(drone|groundstation)_(0i)', where `i` is the index. Default: all locations",
            action="append",
            default=argparse.SUPPRESS,
        )
        parser.add_argument("--online", help="Record online monitoring performance", action="store_true")
        parser.add_argument("--num-trials", help="Number of timing trials to run", default=N_TRIALS, type=int)
        parser.add_argument("--num-repeats", help="Number of repeats per trial", default=N_REPEATS, type=int)
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--timeit", help="Record performance", action="store_true")
        group.add_argument("--profile", help="Profile the monitoring code", action="store_true")

        args = parser.parse_args()
        assert args.spec.is_file(), "Specification file doesn't exist"
        assert args.map.is_file(), "Map file doesn't exist"
        assert args.trace.is_file(), "Trace file doesn't exist"
        return Args(**vars(args))

    @model_validator(mode="after")
    def _path_exists(self) -> Self:
        assert self.spec_file.is_file(), "Specification file does not exist"
        assert self.map_info.is_file(), "Map info file does not exist"
        assert self.trace.is_file(), "Trace file does not exist"
        return self


def read_spec_file(spec_file: Path) -> tuple[strel.Expr, str]:
    """Return the STREL expression defined in the file along with the DIST_ATTR distance attribute for the spec"""
    import importlib.util

    assert spec_file.is_file()

    specification_module_spec = importlib.util.spec_from_file_location("strel_spec", spec_file.absolute())
    assert specification_module_spec is not None
    specification_module = importlib.util.module_from_spec(specification_module_spec)
    assert specification_module_spec.loader is not None
    specification_module_spec.loader.exec_module(specification_module)

    expr = strel.parse(specification_module.SPECIFICATION)
    dist_attr: str = specification_module.DIST_ATTR

    return expr, dist_attr


def _get_distance_to_obstacle(
    trace: MutableSequence[tuple[float, "nx.Graph[str]"]], map_info: Map
) -> MutableSequence[tuple[float, "nx.Graph[str]"]]:
    """Iterate over the graphs and, to each drone location, add a "dist_to_obstacle" parameter"""
    from scipy.spatial.distance import cdist

    # Unpack obstacles
    obstacle_centers, obstacle_ids = zip(*[((obs.x, obs.y), obs.id) for obs in map_info.buildings])
    assert len(obstacle_centers) == len(obstacle_centers)
    obstacle_centers = np.array(obstacle_centers)
    obstacle_ids = list(obstacle_ids)
    assert obstacle_centers.shape == (len(obstacle_centers), 2)
    obstacle_radius = map_info.map_properties.building_width

    for _, sample in trace:
        # Unpack locations
        loc_centers, loc_ids = zip(
            *[
                ((dr_data["pos_x"], dr_data["pos_y"]), dr)
                for dr, dr_data in sample.nodes(data=True)
                if dr_data["kind"] == "drone"
            ]
        )
        loc_centers = np.array(loc_centers)
        loc_ids = list(loc_ids)
        assert loc_centers.shape == (len(loc_centers), 2)

        # Compute distance to obstacles
        dist_to_each_obstacle = cdist(loc_centers, obstacle_centers, "euclidean")
        assert dist_to_each_obstacle.shape == (len(loc_ids), len(obstacle_ids))
        # Compute min dist to obstacles (along each row)
        # Subtract the radius and max by 0
        min_dist_to_obstacle = np.maximum(
            np.absolute((np.amin(dist_to_each_obstacle, axis=1) - obstacle_radius)),
            np.array(0.0),
        )
        assert min_dist_to_obstacle.shape == (len(loc_ids),)

        # For each location, add the dist_to_obstacle attribute
        for i, dr in enumerate(loc_ids):
            sample.nodes[dr]["dist_to_obstacle"] = min_dist_to_obstacle[i]
    return trace


def read_trace(trace_file: Path, map_info: Map) -> Sequence[tuple[float, nx.Graph]]:
    """Convert raw trace into dynamic graph signals"""
    # Read the trace file as a csv
    with open(trace_file, "r") as f:
        reader = csv.DictReader(f)
        raw_trace = [TraceSample.model_validate(row) for row in reader]

    goal_pos = np.array([map_info.goal_position.x, map_info.goal_position.y], dtype=np.float64)

    # Convert the map info GCSs into a graph too
    gcs_graph: "nx.Graph[str]" = nx.Graph()
    for gcs in map_info.ground_stations:
        # Compute distance to goal.
        gcs_pos = np.array([gcs.x, gcs.y])
        dist_sq = np.sum(np.square(goal_pos - gcs_pos))
        gcs_graph.add_node(
            f"gcs_{gcs.id:02d}",
            kind="groundstation",
            pos_x=gcs.x,
            pos_y=gcs.y,
            dist_to_goal=np.sqrt(dist_sq),
            dist_to_obstacle=0.0,
        )

    for d1, d2 in itertools.combinations(gcs_graph.nodes, 2):
        x1, y1 = gcs_graph.nodes[d1]["pos_x"], gcs_graph.nodes[d1]["pos_y"]
        x2, y2 = gcs_graph.nodes[d2]["pos_x"], gcs_graph.nodes[d2]["pos_y"]
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if dist <= map_info.map_properties.radius_of_influence:
            gcs_graph.add_edge(
                d1,
                d2,
                hop=1,
                dist=dist,
            )

    # For each timestamp, create a nx.Graph
    prev_time: float | None = None
    trace: deque[tuple[float, "nx.Graph[str]"]] = deque()
    for sample in raw_trace:
        if prev_time is None or prev_time < sample.time:
            # Detected new time point. Create a new graph
            trace.append((sample.time, nx.Graph()))
            prev_time = sample.time
            pass
        # Add current row to last graph
        g = trace[-1][1]
        # Compute distance to goal.
        drone_pos = np.array([sample.pos_x, sample.pos_y])
        dist_sq = np.sum(np.square(goal_pos - drone_pos))
        g.add_node(
            f"drone_{sample.drone:02d}",
            kind="drone",
            pos_x=sample.pos_x,
            pos_y=sample.pos_y,
            dist_to_goal=np.sqrt(dist_sq),
        )

    # Add an edge between drones if they are within communication distance.
    for _, g in trace:
        for d1, d2 in itertools.combinations(g.nodes, 2):
            x1, x2 = g.nodes[d1]["pos_x"], g.nodes[d2]["pos_x"]
            y1, y2 = g.nodes[d1]["pos_y"], g.nodes[d2]["pos_y"]
            dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist <= DRONE_COMMS_RADIUS:
                g.add_edge(d1, d2, hop=1, dist=dist)

        # Connect a GCS to a drone if in radius
        new_edges = []
        for gcs, drone in itertools.product(gcs_graph.nodes, g.nodes):
            x1, y1 = gcs_graph.nodes[gcs]["pos_x"], gcs_graph.nodes[gcs]["pos_y"]
            x2, y2 = g.nodes[drone]["pos_x"], g.nodes[drone]["pos_y"]
            dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist <= map_info.map_properties.radius_of_influence:
                new_edges.append((gcs, drone, dict(hop=1, dist=dist)))

        # Merge the gcs graph and the drone graph
        g.update(gcs_graph)
        g.add_edges_from(new_edges)

    return trace


def assign_bool_labels(input: Alph, loc: Location, pred: str) -> bool:  # noqa: N802
    match pred:
        case "drone" | "groundstation":
            return input.nodes[loc]["kind"] == pred
        case "obstacle":
            return input.nodes[loc]["dist_to_obstacle"] <= CLOSE_TO_OBSTACLE_THRESH
        case "goal":
            return input.nodes[loc]["dist_to_goal"] <= CLOSE_TO_GOAL_THRESH

    ret = input.nodes[loc][pred]
    assert isinstance(ret, bool)
    return ret


def online_monitoring(
    args: Args,
    monitor: StrelAutomaton,
    trace: Sequence["nx.Graph[Location]"],
    ego_locs: Mapping[str, int],
) -> None:
    if args.timeit:
        print(
            f"""
Logging online monitoring performance.

In the output, there are three fields for each ego location. 

* The trace length.
* The repetition count (‘best of 5’) which tells you how many times the trace was monitored.
* The time monitoring took on average per step on within the best repetition. That is, the average per step performance for the best trace performance.

Finally, the average best per step time is reported across all ego locations

Monitoring for ego locations: {list(ego_locs.keys())}


"""
        )
        overall_results = []
        for ego, ego_loc in ego_locs.items():
            timer = timeit.Timer(lambda ego=ego, ego_loc=ego_loc: forward_run(monitor, trace, {ego: ego_loc}), "gc.enable()")
            results = timer.repeat(repeat=args.num_repeats, number=1)
            step_results = map(lambda res: res / len(trace), results)
            best_time = min(step_results)
            overall_results.append(best_time)
            print(
                f"phi @ {ego:15s} ==> best of {args.num_repeats} monitoring runs on trace of length {len(trace)}: {best_time:.6e} sec per step"
            )

        print()
        print(f"==> Overall average: {sum(overall_results) / len(overall_results):.6e} sec per step")
    else:
        print("Begin monitoring trace")
        start_time = perf_counter_ns()
        check = forward_run(monitor, trace, ego_locs)
        end_time = perf_counter_ns()
        t_delta = (end_time - start_time) * 10e-9
        print(f"Completed monitoring in: \t\t{t_delta:.6e} seconds")
        print(f"Average per step time: \t\t{t_delta / len(trace)} seconds")
        print()
        for name, sat in check.items():
            print(f"\tphi @ {name} = {sat}")


def forward_run(
    monitor: StrelAutomaton,
    trace: Sequence["nx.Graph[Location]"],
    ego_locs: Mapping[str, int],
) -> dict[str, bool]:
    states = {name: monitor.initial_at(loc) for name, loc in ego_locs.items()}
    for input in trace:
        states = {name: monitor.next(input, state) for name, state in states.items()}
    return {name: monitor.final_weight(state) for name, state in states.items()}


def offline_monitoring(
    args: Args,
    monitor: StrelAutomaton,
    trace: Sequence["nx.Graph[Location]"],
    ego_locs: Mapping[str, int],
) -> None:
    if args.timeit:
        print(
            f"""
Logging offline monitoring performance.

In the output, there are three fields. 

* The loop count, which is number of times the trace was monitored per timing loop repition
* The repetition count (‘best of 5’) which tells you how many times the timing loop was repeated
* Time the monitoring took on average within the best repetition of the timing loop. That is, the time the fastest repetition took divided by the loop count.

Monitoring for ego locations: {list(ego_locs.keys())}


"""
        )
        timer = timeit.Timer(lambda: forward_run(monitor, trace, ego_locs), "gc.enable()")
        results = [0.0] * args.num_trials
        for i in range(args.num_trials):
            best_repeat = min(timer.repeat(args.num_repeats, 1))
            results[i] = best_repeat

        avg_time = sum(results) / len(results)
        print(f"==> best of {args.num_repeats} runs, avg. of {args.num_trials} trials: {avg_time} sec per run")
    else:
        print("Begin monitoring trace")
        start_time = perf_counter_ns()
        check = forward_run(monitor, trace, ego_locs)
        end_time = perf_counter_ns()
        t_delta = (end_time - start_time) * 10e-9
        print(f"Completed monitoring in: \t\t{t_delta:.6e} seconds")
        print()
        for name, sat in check.items():
            print(f"\tphi @ {name} = {sat}")


def profile_monitoring(
    args: Args,
    monitor: StrelAutomaton,
    trace: Sequence["nx.Graph[Location]"],
    ego_locs: Mapping[str, int],
) -> None:
    with cProfile.Profile() as pr:
        for _ in range(args.num_trials):
            forward_run(monitor, trace, ego_locs)
        pr.print_stats()
        pr.dump_stats(str(Path(__file__).with_suffix(".prof")))


def main(args: Args) -> None:
    print("================================================================================")
    # print(
    #     """
    # WARNING:
    # Longer traces will take time to run due to pre-calculations needed to make
    # the dynamic graphs. This does not measure the actual time it takes to
    # monitor things (which will be timed and reported).
    # """
    # )
    print(f"Reading specification from: {str(args.spec_file)}")
    print(f"Reading map from: {str(args.map_info)}")
    print(f"Reading trace from: {str(args.trace)}")
    print()

    spec, dist_attr = read_spec_file(args.spec_file)
    print(f"phi = {str(spec)}")
    print()
    with open(args.map_info, "r") as f:
        map_info = Map.model_validate(json.load(f))
    trace = list(read_trace(args.trace, map_info))
    print(f"Trace Length  = {len(trace)}")
    max_locs = max([g.number_of_nodes() for _, g in trace])
    print(f"Num Locations = {max_locs}")

    trace = _get_distance_to_obstacle(trace, map_info)
    # Remove timestamps from trace, and relabel the traces with integer nodes
    remapping = {name: i for i, name in enumerate(trace[0][1].nodes)}
    new_trace: list["nx.Graph[int]"] = [nx.relabel_nodes(g, remapping) for _, g in trace]  # type: ignore
    assert len(new_trace) == len(trace)
    assert isinstance(new_trace[0], nx.Graph)

    monitor = make_bool_automaton(
        spec,
        assign_bool_labels,
        dist_attr,
    )
    ego_locs = {ego: remapping[ego] for ego in args.ego_loc}
    if len(ego_locs) == 0:
        ego_locs = remapping

    if args.profile:
        profile_monitoring(args, monitor, new_trace, ego_locs)
    else:
        if args.online:
            online_monitoring(args, monitor, new_trace, ego_locs)
        else:
            offline_monitoring(args, monitor, new_trace, ego_locs)

    print("================================================================================")
    # dd.BDD prints a bunch of errors because of refcounting errors, but we don't care coz the OS will take care of that.
    # original_stderr = sys.stderr
    # original_stdout = sys.stdout
    # devnull = open(os.devnull, "w")
    # sys.stdout = devnull
    # sys.stderr = devnull


if __name__ == "__main__":
    main(Args.parse())
