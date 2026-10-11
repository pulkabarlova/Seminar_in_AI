"""Random obstacle maps for the path-planning comparison.
"""

import json
import math
import random

from polygenerator import random_convex_polygon, random_star_shaped_polygon
from shapely.geometry import LineString, Point, Polygon

MAP_SIZE = 10
START = (0, 0)
END = (10, 10)



def place_polygon(unit_points, center_x, center_y, radius):
    """Move a polygon from the unit square [0, 1] x [0, 1] so it is centred at
    (center_x, center_y) and fits in a square of side 2 * radius."""
    return [
        (center_x + (x - 0.5) * 2 * radius, center_y + (y - 0.5) * 2 * radius)
        for x, y in unit_points
    ]


def generate_simple_obstacle(center_x, center_y, radius=1.0):
    """Random convex polygon with 3-5 vertices."""
    return place_polygon(random_convex_polygon(random.randint(3, 5)), center_x, center_y, radius)


def generate_complex_obstacle(center_x, center_y, radius=1.0):
    """Random star-shaped (usually concave) polygon with 5-8 vertices."""
    return place_polygon(random_star_shaped_polygon(random.randint(5, 8)), center_x, center_y, radius)



def generate_map(
    obstacle_generator,
    n_obstacle=None,
    map_size=MAP_SIZE,
    min_distance=2.5,
    margin=1,
    gap=0.1,
    start=START,
    end=END,
    clear_radius=0.5,
    keep_line_clear=False,
    attempts_limit=1000,
):
    """Generate one map as a list of obstacles (each a list of (x, y) vertices).

    A candidate obstacle is rejected and re-drawn if:
    - its centre is closer than `min_distance` to another obstacle's centre,
    - its shape overlaps or comes closer than `gap` to another obstacle,
    - it covers the start or goal point (within `clear_radius`),
    - `keep_line_clear=True` and it touches the straight start-goal line.
    """
    obstacle_count = n_obstacle if n_obstacle is not None else random.randint(2, 7)

    protected = [Point(start).buffer(clear_radius), Point(end).buffer(clear_radius)]
    if keep_line_clear:
        protected.append(LineString([start, end]).buffer(clear_radius))

    obstacles, shapes, centers = [], [], []
    for _ in range(obstacle_count):
        for _ in range(attempts_limit):
            cx = random.uniform(margin, map_size - margin)
            cy = random.uniform(margin, map_size - margin)
            if any(math.dist((cx, cy), c) < min_distance for c in centers):
                continue

            radius = random.uniform(0.6, 1.5)
            obstacle = obstacle_generator(cx, cy, radius)
            shape = Polygon(obstacle)

            if not shape.is_valid:
                continue
            if any(shape.distance(other) < gap for other in shapes):
                continue
            if any(shape.intersects(area) for area in protected):
                continue

            obstacles.append(obstacle)
            shapes.append(shape)
            centers.append((cx, cy))
            break
        else:
            print(f"WARNING: could only place {len(obstacles)}/{obstacle_count} obstacles")
            break
    return obstacles


def generate_dense_scaling_map(
    obstacle_count,
    map_size=40,
    start=(2, 2),
    end=(38, 38),
    obstacle_radius=(0.35, 0.75),
    min_distance=0.85,
    gap=0.1,
    corridor_width=None,
    clear_radius=1.5,
    obstacle_generator=generate_simple_obstacle,
    seed=None,
    attempts_per_obstacle=300,
):
    """Large map with many small obstacles, for runtime-vs-obstacle-count tests.

    Obstacles don't overlap (their shapes stay at least `gap` apart), stay
    inside the map border and keep `clear_radius` away from start and goal,
    so the free space is connected and a path always exists.

    No free corridor by default, so the straight start-goal line is usually
    blocked. Pass `corridor_width` (e.g. 1.2) to keep a free corridor of that
    half-width along the straight line, as in the old version.

    `seed` makes the map reproducible without changing the global random
    state for the code that runs afterwards.
    """
    saved_state = random.getstate()
    if seed is not None:
        random.seed(seed)

    try:
        protected = [Point(start).buffer(clear_radius), Point(end).buffer(clear_radius)]
        if corridor_width:
            protected.append(LineString([start, end]).buffer(corridor_width))

        obstacles, shapes, centers = [], [], []
        attempts_limit = obstacle_count * attempts_per_obstacle
        attempts = 0

        while len(obstacles) < obstacle_count and attempts < attempts_limit:
            attempts += 1
            radius = random.uniform(*obstacle_radius)
            cx = random.uniform(1 + radius, map_size - 1 - radius)
            cy = random.uniform(1 + radius, map_size - 1 - radius)

            if any(math.dist((cx, cy), c) < min_distance for c in centers):
                continue
            obstacle = obstacle_generator(cx, cy, radius)
            shape = Polygon(obstacle)
            if not shape.is_valid:
                continue
            if any(shape.intersects(area) for area in protected):
                continue
            if any(shape.distance(other) < gap for other in shapes):
                continue

            obstacles.append(obstacle)
            shapes.append(shape)
            centers.append((cx, cy))
    finally:
        if seed is not None:
            random.setstate(saved_state)

    if len(obstacles) < obstacle_count:
        print(f"WARNING: placed {len(obstacles)}/{obstacle_count} obstacles")

    return obstacles



def check_map(obstacles, start=START, end=END):
    """Return (number of overlapping obstacle pairs, whether start or goal is covered)."""
    shapes = [Polygon(p) for p in obstacles]
    overlaps = sum(
        shapes[i].intersects(shapes[j])
        for i in range(len(shapes))
        for j in range(i + 1, len(shapes))
    )
    blocked = any(s.covers(Point(start)) or s.covers(Point(end)) for s in shapes)
    return overlaps, blocked


def save_batches(path, batches, seed=None, map_size=MAP_SIZE, start=START, end=END):
    """Save batches for further reuse."""
    with open(path, "w") as f:
        json.dump(
            {
                "seed": seed,
                "map_size": map_size,
                "start": start,
                "end": end,
                "batches": {f"batch{i + 1}": batch for i, batch in enumerate(batches)},
            },
            f,
        )


def load_batches(path):
    """Return the list of batches saved by `save_batches`."""
    with open(path) as f:
        return list(json.load(f)["batches"].values())
