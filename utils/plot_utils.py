"""Plotting helpers: obstacle maps, planner paths and benchmark charts."""

import math

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from algorithms.PRM import PRM
from algorithms.RRT_star import RRTStar
from utils.compare_tables_util import ALGORITHMS, algorithm_batch_means
from utils.path_planning_utils import (
    _to_xy,
    build_vg_graph,
    find_shortest_route_VG,
    prepare_shapely_obstacles,
    prepare_vg_obstacles,
    prepare_vg_point,
)

ALGORITHM_COLORS = {
    "VG": "#F28B9B",
    "RRT_star": "#A78BDD",
    "PRM": "#6DBF8A",
}


def plot_map(ax, obstacles, map_size=10, start=(0, 0), end=(10, 10), show_line=False, title=None):
    """Draw one map on `ax`. Gray: obstacles, green: start, blue: goal,
    red dashed (optional): straight start-goal line."""
    for poly in obstacles:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        ax.fill(xs, ys, color="gray", alpha=0.7)
        ax.plot(xs + [xs[0]], ys + [ys[0]], "k-", linewidth=1)
    if show_line:
        ax.plot([start[0], end[0]], [start[1], end[1]], "r--", linewidth=1)
    ax.scatter(*start, color="green", zorder=3)
    ax.scatter(*end, color="blue", zorder=3)
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)


def plot(batch, title=None, map_size=10, cols=5, start=(0, 0), end=(10, 10), show_line=False):
    """Draw every map of a batch in a grid."""
    if not batch:
        return
    cols = min(cols, len(batch))
    rows = math.ceil(len(batch) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(batch):
            ax.axis("off")
            continue
        plot_map(
            ax, batch[i], map_size=map_size, start=start, end=end,
            show_line=show_line, title=f"Map {i + 1}: {len(batch[i])} obs",
        )

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------------- PLANNER PATHS ----------------

def _plot_obstacles(ax, obstacles):
    for polygon in obstacles:
        if not polygon:
            continue

        if isinstance(polygon, Polygon):
            coords = list(polygon.exterior.coords)
            xs = [point[0] for point in coords]
            ys = [point[1] for point in coords]
        else:
            xs = [point[0] for point in polygon] + [polygon[0][0]]
            ys = [point[1] for point in polygon] + [polygon[0][1]]

        ax.fill(xs, ys, color="lightgray", alpha=0.7)
        ax.plot(xs, ys, "k-")


def _plot_path(ax, path):
    if not path:
        return
    path_xy = [_to_xy(point) for point in path]
    ax.plot(
        [point[0] for point in path_xy],
        [point[1] for point in path_xy],
        "r-",
        linewidth=2,
    )


def _plot_start_end(ax, start, end):
    start_x, start_y = _to_xy(start)
    end_x, end_y = _to_xy(end)
    ax.scatter([start_x], [start_y], color="green", label="start")
    ax.scatter([end_x], [end_y], color="blue", label="end")


def _prepare_axes(ax=None, title=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    return ax


def plot_shortest_route_VG(start, end, obstacles, graph, ax=None):
    path = find_shortest_route_VG(start, end, graph)

    ax = _prepare_axes(ax=ax, title="VG shortest path")
    _plot_obstacles(ax, obstacles)

    for edge in graph.visgraph.edges:
        ax.plot([edge.p1.x, edge.p2.x], [edge.p1.y, edge.p2.y], color="gray", linewidth=0.5)

    _plot_path(ax, path)
    _plot_start_end(ax, start, end)
    return path, ax


def plot_shortest_route_RRT_star(
    start,
    end,
    obstacles,
    iterations=2000,
    step=0.5,
    gamma=2.0,
    goal_sample_rate=0.1,
    ax=None,
):
    planner = RRTStar(
        start,
        end,
        obstacles,
        step=step,
        gamma=gamma,
        goal_sample_rate=goal_sample_rate,
    )
    path = planner.plan(iterations=iterations)

    ax = _prepare_axes(ax=ax, title="RRT* shortest path")
    _plot_obstacles(ax, obstacles)

    for node in planner.nodes:
        if node.parent is None:
            continue
        ax.plot(
            [node.point[0], node.parent.point[0]],
            [node.point[1], node.parent.point[1]],
            color="gray",
            linewidth=0.5,
        )

    _plot_path(ax, path)
    _plot_start_end(ax, start, end)
    return path, ax


def plot_shortest_route_PRM(
    start,
    end,
    obstacles,
    rand_area=(-1, 10),
    n_samples=200,
    k=10,
    max_edge_len=None,
    ax=None,
):
    planner = PRM(obstacles, rand_area=rand_area, n_samples=n_samples, k=k, max_edge_len=max_edge_len)
    path = planner.plan(start, end)

    ax = _prepare_axes(ax=ax, title="PRM shortest path")
    _plot_obstacles(ax, obstacles)

    for p1, p2 in planner.graph.edges:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", linewidth=0.5)

    _plot_path(ax, path)
    _plot_start_end(ax, start, end)
    return path, ax


def plot_batch_examples_VG(batches, start, end, n_examples=3):
    return _plot_batch_examples(
        batches,
        start,
        end,
        "VG",
        _plot_vg_case,
        n_examples=n_examples,
    )


def plot_batch_examples_RRT_star(batches, start, end, n_examples=3, **rrt_kwargs):
    return _plot_batch_examples(
        batches,
        start,
        end,
        "RRT_star",
        _plot_rrt_case,
        n_examples=n_examples,
        planner_kwargs=rrt_kwargs,
    )


def plot_batch_examples_PRM(batches, start, end, n_examples=3, **prm_kwargs):
    return _plot_batch_examples(
        batches,
        start,
        end,
        "PRM",
        _plot_prm_case,
        n_examples=n_examples,
        planner_kwargs=prm_kwargs,
    )


def _plot_batch_examples(
    batches,
    start,
    end,
    algorithm_name,
    plotter,
    n_examples=3,
    planner_kwargs=None,
):
    planner_kwargs = planner_kwargs or {}
    figures = []

    for batch_idx, batch in enumerate(batches, start=1):
        example_count = min(n_examples, len(batch))
        fig, axes = plt.subplots(1, example_count, figsize=(6 * example_count, 6))
        if example_count == 1:
            axes = [axes]

        fig.suptitle(f"{algorithm_name}: batch_{batch_idx}", fontsize=14)

        for map_idx in range(example_count):
            ax = axes[map_idx]
            obstacles = batch[map_idx]

            try:
                plotter(start, end, obstacles, ax=ax, planner_kwargs=planner_kwargs)
                ax.set_title(f"map_{map_idx + 1}")
            except Exception as exc:
                _plot_obstacles(ax, obstacles)
                _plot_start_end(ax, start, end)
                ax.set_aspect("equal")
                ax.set_title(f"map_{map_idx + 1}: error")
                ax.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax.transAxes)

        plt.tight_layout()
        figures.append(fig)

    return figures


def _plot_vg_case(start, end, obstacles, ax, planner_kwargs):
    del planner_kwargs
    vg_start = prepare_vg_point(start)
    vg_end = prepare_vg_point(end)
    vg_obstacles = prepare_vg_obstacles(obstacles)
    vg_graph = build_vg_graph(vg_obstacles)
    plot_shortest_route_VG(vg_start, vg_end, obstacles, vg_graph, ax=ax)


def _plot_rrt_case(start, end, obstacles, ax, planner_kwargs):
    shapely_obstacles = prepare_shapely_obstacles(obstacles)
    plot_shortest_route_RRT_star(start, end, shapely_obstacles, ax=ax, **planner_kwargs)


def _plot_prm_case(start, end, obstacles, ax, planner_kwargs):
    shapely_obstacles = prepare_shapely_obstacles(obstacles)
    plot_shortest_route_PRM(start, end, shapely_obstacles, ax=ax, **planner_kwargs)


# ---------------- BENCHMARK CHARTS ----------------

def plot_algorithm_batch_means(
    tables,
    title,
    ylabel,
    algorithms=ALGORITHMS,
    ax=None,
    ymin=None,
):
    columns, means = algorithm_batch_means(tables, algorithms=algorithms)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    x_positions = list(range(len(columns)))
    bar_width = 0.9 / len(algorithms)
    offset_start = -0.4 + bar_width / 2

    for alg_idx, algorithm in enumerate(algorithms):
        positions = [
            position + offset_start + alg_idx * bar_width
            for position in x_positions
        ]
        ax.bar(
            positions,
            means[algorithm],
            width=bar_width,
            label=algorithm,
            color=ALGORITHM_COLORS[algorithm],
        )

    if ymin is not None:
        ax.set_ylim(bottom=ymin)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(columns, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return columns, means, ax


def plot_algorithm_graph_big_data_comparison(
    tables,
    title,
    ylabel,
    algorithms=ALGORITHMS,
    ax=None,
):

    columns, means = algorithm_batch_means(tables, algorithms=algorithms)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    for algorithm in algorithms:
        ax.plot(
            columns,
            means[algorithm],
            marker="o",
            label=algorithm,
            color=ALGORITHM_COLORS[algorithm],
            linewidth=2,
            markersize=5,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("obstacle count")
    ax.legend()
    ax.grid(alpha=0.25)
    return columns, means, ax
