"""Running and benchmarking the planners: Visibility Graph (pyvisgraph), RRT* and PRM.

Tables for the results are in `utils.compare_tables_util`, plots in `utils.plot_utils`.
"""

import math
from time import perf_counter

import pyvisgraph as vg
from shapely.geometry import Polygon

from algorithms.PRM import PRM
from algorithms.RRT_star import RRTStar


def _to_xy(point):
    if hasattr(point, "x") and hasattr(point, "y"):
        return point.x, point.y
    return point[0], point[1]


def _to_vg_point(point):
    if isinstance(point, vg.Point):
        return point
    return vg.Point(point[0], point[1])


def _to_vg_obstacles(obstacles):
    return [
        [_to_vg_point(vertex) for vertex in polygon]
        for polygon in obstacles
        if len(polygon) >= 3
    ]


def _to_shapely_obstacles(obstacles):
    return [Polygon(polygon) for polygon in obstacles if len(polygon) >= 3]


def prepare_vg_point(point):
    return _to_vg_point(point)


def prepare_vg_obstacles(obstacles):
    return _to_vg_obstacles(obstacles)


def prepare_shapely_obstacles(obstacles):
    return _to_shapely_obstacles(obstacles)


def build_prm_roadmap(obstacles, rand_area=(-1, 10), n_samples=200, k=10, max_edge_len=None):
    planner = PRM(obstacles, rand_area=rand_area, n_samples=n_samples, k=k, max_edge_len=max_edge_len)
    planner.build_roadmap()
    return planner


def build_vg_graph(vg_obstacles):
    graph = vg.VisGraph()
    # status=False turns off pyvisgraph's progress bar, which otherwise
    # prints on every build and adds to the measured runtime
    graph.build(vg_obstacles, status=False)
    return graph


def find_shortest_route_VG(start, end, graph):
    return graph.shortest_path(start, end)


def find_shortest_route_RRT_star(
    start,
    end,
    obstacles,
    iterations=2000,
    step=0.5,
    gamma=2.0,
    goal_sample_rate=0.1,
):
    planner = RRTStar(
        start,
        end,
        obstacles,
        step=step,
        gamma=gamma,
        goal_sample_rate=goal_sample_rate,
    )
    return planner.plan(iterations=iterations)


def find_shortest_route_PRM(
    start,
    end,
    obstacles,
    rand_area=(-1, 10),
    n_samples=200,
    k=10,
    max_edge_len=None,
):
    planner = PRM(obstacles, rand_area=rand_area, n_samples=n_samples, k=k, max_edge_len=max_edge_len)
    return planner.plan(start, end)


def find_shortest_route_PRM_reuse(start, end, planner):
    return planner.query(start, end)


def benchmark_batches(
    batches,
    start,
    end,
    rrt_kwargs=None,
    prm_kwargs=None,
    include_algorithm_setup=False,
    runtime_runs=10,
):
    rrt_kwargs = rrt_kwargs or {}
    prm_kwargs = prm_kwargs or {}
    map_count = min(len(batch) for batch in batches)

    vg_results = []
    rrt_results = []
    prm_results = []

    for map_idx in range(map_count):
        vg_row = []
        rrt_row = []
        prm_row = []

        for batch in batches:
            obstacles = batch[map_idx]

            vg_row.append(_benchmark_vg_case(start, end, obstacles, include_algorithm_setup, runtime_runs))
            rrt_row.append(_benchmark_rrt_case(start, end, obstacles, rrt_kwargs, include_algorithm_setup, runtime_runs))
            prm_row.append(_benchmark_prm_case(start, end, obstacles, prm_kwargs, include_algorithm_setup, runtime_runs))

        vg_results.append(vg_row)
        rrt_results.append(rrt_row)
        prm_results.append(prm_row)

    columns = [f"batch_{idx + 1}" for idx in range(len(batches))]
    index = [f"map_{idx + 1}" for idx in range(map_count)]

    return {
        "VG": {"index": index, "columns": columns, "values": vg_results},
        "RRT_star": {"index": index, "columns": columns, "values": rrt_results},
        "PRM": {"index": index, "columns": columns, "values": prm_results},
    }


def benchmark_first_map_fixed_points_reuse(
    batches,
    start,
    end,
    trials=10,
    rrt_kwargs=None,
    prm_kwargs=None,
):
    return _benchmark_first_map_trials(
        batches=batches,
        trials=trials,
        start=start,
        end=end,
        rrt_kwargs=rrt_kwargs,
        prm_kwargs=prm_kwargs,
    )


def build_path_length_tables(
    batches,
    start,
    end,
    rrt_kwargs=None,
    prm_kwargs=None,
    prm_reuse=False,
):
    rrt_kwargs = rrt_kwargs or {}
    prm_kwargs = prm_kwargs or {}
    map_count = min(len(batch) for batch in batches)

    vg_values = []
    rrt_values = []
    prm_values = []

    for map_idx in range(map_count):
        vg_row = []
        rrt_row = []
        prm_row = []

        for batch in batches:
            obstacles = batch[map_idx]
            vg_row.append(_path_length_vg(start, end, obstacles))
            rrt_row.append(_path_length_rrt(start, end, obstacles, rrt_kwargs))
            prm_row.append(_path_length_prm(start, end, obstacles, prm_kwargs, prm_reuse))

        vg_values.append(vg_row)
        rrt_values.append(rrt_row)
        prm_values.append(prm_row)

    columns = [f"batch_{idx + 1}" for idx in range(len(batches))]
    index = [f"map_{idx + 1}" for idx in range(map_count)]
    return {
        "VG": {"index": index, "columns": columns, "values": vg_values},
        "RRT_star": {"index": index, "columns": columns, "values": rrt_values},
        "PRM": {"index": index, "columns": columns, "values": prm_values},
    }


def _measure_runtime(func, *args, runs=10, **kwargs):
    times = []
    for _ in range(runs):
        started_at = perf_counter()
        try:
            func(*args, **kwargs)
        except Exception:
            return float("nan")
        times.append(perf_counter() - started_at)
    return sum(times) / len(times)


def _benchmark_first_map_trials(
    batches,
    trials,
    start,
    end,
    rrt_kwargs=None,
    prm_kwargs=None,
):
    rrt_kwargs = rrt_kwargs or {}
    prm_kwargs = prm_kwargs or {}

    vg_contexts = []
    prm_contexts = []

    for batch in batches:
        obstacles = batch[0]
        vg_contexts.append(_prepare_vg_context(obstacles))
        prm_contexts.append(_prepare_prm_context(obstacles, prm_kwargs))

    vg_values = []
    rrt_values = []
    prm_values = []

    for trial_idx in range(trials):
        vg_row = []
        rrt_row = []
        prm_row = []

        for batch_idx, batch in enumerate(batches):
            obstacles = batch[0]
            start_point, end_point = start, end

            if not _points_valid_for_obstacles(start_point, end_point, obstacles):
                vg_row.append(float("nan"))
                rrt_row.append(float("nan"))
                prm_row.append(float("nan"))
                continue

            vg_row.append(_benchmark_vg_reuse_case(start_point, end_point, vg_contexts[batch_idx]))
            prm_row.append(_benchmark_prm_reuse_case(start_point, end_point, prm_contexts[batch_idx]))
            rrt_row.append(_benchmark_rrt_case(start_point, end_point, obstacles, rrt_kwargs, True))

        vg_values.append(vg_row)
        rrt_values.append(rrt_row)
        prm_values.append(prm_row)

    columns = [f"batch_{idx + 1}" for idx in range(len(batches))]
    index = [f"trial_{idx + 1}" for idx in range(trials)]
    return {
        "VG": {"index": index, "columns": columns, "values": vg_values},
        "RRT_star": {"index": index, "columns": columns, "values": rrt_values},
        "PRM": {"index": index, "columns": columns, "values": prm_values},
    }


def _benchmark_vg_case(start, end, obstacles, include_algorithm_setup, runtime_runs=10):
    try:
        vg_start = prepare_vg_point(start)
        vg_end = prepare_vg_point(end)
        vg_obstacles = prepare_vg_obstacles(obstacles)
    except Exception:
        return float("nan")

    if include_algorithm_setup:
        return _measure_runtime(_run_vg_full, vg_start, vg_end, vg_obstacles, runs=runtime_runs)

    try:
        vg_graph = build_vg_graph(vg_obstacles)
    except Exception:
        return float("nan")

    return _measure_runtime(find_shortest_route_VG, vg_start, vg_end, vg_graph, runs=runtime_runs)


def _benchmark_rrt_case(start, end, obstacles, rrt_kwargs, include_algorithm_setup, runtime_runs=10):
    try:
        shapely_obstacles = prepare_shapely_obstacles(obstacles)
    except Exception:
        return float("nan")

    if include_algorithm_setup:
        return _measure_runtime(_run_rrt_full, start, end, shapely_obstacles, rrt_kwargs, runs=runtime_runs)

    return _measure_runtime(
        find_shortest_route_RRT_star,
        start,
        end,
        shapely_obstacles,
        runs=runtime_runs,
        **rrt_kwargs,
    )


def _benchmark_prm_case(start, end, obstacles, prm_kwargs, include_algorithm_setup, runtime_runs=10):
    try:
        shapely_obstacles = prepare_shapely_obstacles(obstacles)
    except Exception:
        return float("nan")

    if include_algorithm_setup:
        return _measure_runtime(_run_prm_full, start, end, shapely_obstacles, prm_kwargs, runs=runtime_runs)

    return _measure_runtime(
        find_shortest_route_PRM,
        start,
        end,
        shapely_obstacles,
        runs=runtime_runs,
        **prm_kwargs,
    )


def _benchmark_vg_reuse_case(start, end, vg_context):
    if vg_context is None:
        return float("nan")
    vg_start = prepare_vg_point(start)
    vg_end = prepare_vg_point(end)
    return _measure_runtime(find_shortest_route_VG, vg_start, vg_end, vg_context["graph"])


def _benchmark_prm_reuse_case(start, end, prm_context):
    if prm_context is None:
        return float("nan")
    return _measure_runtime(find_shortest_route_PRM_reuse, start, end, prm_context["planner"])


def _run_vg_full(start, end, vg_obstacles):
    vg_graph = build_vg_graph(vg_obstacles)
    return find_shortest_route_VG(start, end, vg_graph)


def _run_rrt_full(start, end, shapely_obstacles, rrt_kwargs):
    return find_shortest_route_RRT_star(start, end, shapely_obstacles, **rrt_kwargs)


def _run_prm_full(start, end, shapely_obstacles, prm_kwargs):
    return find_shortest_route_PRM(start, end, shapely_obstacles, **prm_kwargs)


def _prepare_vg_context(obstacles):
    try:
        vg_obstacles = prepare_vg_obstacles(obstacles)
        graph = build_vg_graph(vg_obstacles)
        return {"graph": graph}
    except Exception:
        return None


def _prepare_prm_context(obstacles, prm_kwargs):
    try:
        shapely_obstacles = prepare_shapely_obstacles(obstacles)
        planner = build_prm_roadmap(shapely_obstacles, **prm_kwargs)
        return {"planner": planner}
    except Exception:
        return None


def path_length(path):
    if not path or len(path) < 2:
        return float("nan")
    total = 0.0
    path_xy = [_to_xy(point) for point in path]
    for p1, p2 in zip(path_xy, path_xy[1:]):
        total += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    return total


def _points_valid_for_obstacles(start, end, obstacles):
    shapely_obstacles = prepare_shapely_obstacles(obstacles)
    return _point_is_free(start, shapely_obstacles) and _point_is_free(end, shapely_obstacles)


def _point_is_free(point, shapely_obstacles):
    point_geom = _point_geom(point)
    return not any(obs.contains(point_geom) for obs in shapely_obstacles)


def _path_length_vg(start, end, obstacles):
    try:
        if not _points_valid_for_obstacles(start, end, obstacles):
            return float("nan")
        vg_start = prepare_vg_point(start)
        vg_end = prepare_vg_point(end)
        vg_graph = build_vg_graph(prepare_vg_obstacles(obstacles))
        return path_length(find_shortest_route_VG(vg_start, vg_end, vg_graph))
    except Exception:
        return float("nan")


def _path_length_rrt(start, end, obstacles, rrt_kwargs):
    try:
        if not _points_valid_for_obstacles(start, end, obstacles):
            return float("nan")
        path = find_shortest_route_RRT_star(start, end, prepare_shapely_obstacles(obstacles), **rrt_kwargs)
        return path_length(path)
    except Exception:
        return float("nan")


def _path_length_prm(start, end, obstacles, prm_kwargs, prm_reuse):
    try:
        if not _points_valid_for_obstacles(start, end, obstacles):
            return float("nan")
        shapely_obstacles = prepare_shapely_obstacles(obstacles)
        if prm_reuse:
            planner = build_prm_roadmap(shapely_obstacles, **prm_kwargs)
            path = find_shortest_route_PRM_reuse(start, end, planner)
        else:
            path = find_shortest_route_PRM(start, end, shapely_obstacles, **prm_kwargs)
        return path_length(path)
    except Exception:
        return float("nan")


def _point_geom(point):
    from shapely.geometry import Point
    return Point(point)
