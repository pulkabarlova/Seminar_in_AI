import math

import numpy as np
from shapely.geometry import LineString
from scipy.spatial import KDTree

"""
RRT* planner adapted from the PythonRobotics ProbabilisticRoadMap example:
https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/ProbabilisticRoadMap/probabilistic_road_map.py
"""

# ---------------- NODE ----------------

class Node:
    def __init__(self, point):
        self.point = np.array(point, dtype=float)
        self.parent = None
        self.cost = 0.0
        self.children: list["Node"] = []

    @property
    def x(self):
        return self.point[0]

    @property
    def y(self):
        return self.point[1]


# ---------------- RRT* ----------------

class RRTStar:
    def __init__(self, start, goal, obstacles,
                 step=0.5,
                 gamma=2.0,
                 goal_sample_rate=0.1):

        self.start = Node(start)
        self.goal = Node(goal)

        self.step = step
        self.gamma = gamma
        self.goal_sample_rate = goal_sample_rate

        self.nodes = [self.start]
        self.kdtree = self._build_kdtree()

        self.obstacles = obstacles
        self._sample_min, self._sample_max = self._compute_sampling_bounds()

    def plan(self, iterations=2000):
        for _ in range(iterations):
            rnd = self.sample()
            nearest = self.nearest(rnd)
            new_point = self.steer(nearest.point, rnd)

            if new_point is None:
                continue

            new_node = Node(new_point)

            if not self.collision_free(nearest.point, new_node.point):
                continue

            new_node.parent = nearest
            new_node.cost = nearest.cost + self.dist(nearest.point, new_node.point)

            near_nodes = self.near(new_node.point)
            self.choose_parent(new_node, near_nodes)

            if new_node.parent is not None:
                new_node.parent.children.append(new_node)

            self.nodes.append(new_node)
            self.kdtree = self._build_kdtree()

            self.rewire(new_node, near_nodes)

            if self.dist(new_node.point, self.goal.point) < self.step:
                if self.collision_free(new_node.point, self.goal.point):
                    if self.goal.parent is not None:
                        try:
                            self.goal.parent.children.remove(self.goal)
                        except ValueError:
                            pass

                    self.goal.parent = new_node
                    self.goal.cost = new_node.cost + self.dist(new_node.point, self.goal.point)
                    new_node.children.append(self.goal)

                    if self.goal not in self.nodes:
                        self.nodes.append(self.goal)

                    return self.extract_path()

        return self.extract_path()

    # ---------------- SAMPLING ----------------

    def sample(self):
        if np.random.rand() < self.goal_sample_rate:
            return self.goal.point

        return np.random.uniform(self._sample_min, self._sample_max)

    # ---------------- TREE OPS ----------------

    def nearest(self, point):
        _, index = self.kdtree.query(np.asarray(point, dtype=float))
        return self.nodes[int(index)]

    def near(self, point):
        n = len(self.nodes)
        if n < 2:
            return [self.start]

        radius = self.gamma * (math.log(n + 1) / (n + 1)) ** 0.5
        indexes = self.kdtree.query_ball_point(np.asarray(point, dtype=float), radius)
        return [self.nodes[int(index)] for index in indexes]

    # ---------------- MOTION ----------------

    def steer(self, from_point, to_point):
        direction = to_point - from_point
        dist = np.linalg.norm(direction)

        if dist < 1e-9:
            return None

        step = min(self.step, dist)
        return from_point + direction / dist * step

    def dist(self, a, b):
        return np.linalg.norm(a - b)

    def _build_kdtree(self):
        return KDTree([node.point for node in self.nodes])

    # ---------------- COST / RRT* ----------------

    def choose_parent(self, new_node, near_nodes):
        best_cost = new_node.cost
        best_parent = new_node.parent

        for node in near_nodes:
            cost = node.cost + self.dist(node.point, new_node.point)
            if cost < best_cost and self.collision_free(node.point, new_node.point):
                best_cost = cost
                best_parent = node

        new_node.parent = best_parent
        new_node.cost = best_cost

    def rewire(self, new_node, near_nodes):
        for node in near_nodes:
            if node is new_node.parent:
                continue

            new_cost = new_node.cost + self.dist(new_node.point, node.point)

            if new_cost < node.cost and self.collision_free(new_node.point, node.point):
                if node.parent is not None:
                    try:
                        node.parent.children.remove(node)
                    except ValueError:
                        pass

                node.parent = new_node
                new_node.children.append(node)
                self._propagate_cost(node, new_cost)

    def _propagate_cost(self, node, new_cost):
        delta = new_cost - node.cost
        if abs(delta) < 1e-12:
            return

        node.cost = new_cost

        for child in node.children:
            self._propagate_cost(child, child.cost + delta)

    # ---------------- COLLISION ----------------

    def collision_free(self, p1, p2):
        line = LineString([tuple(p1), tuple(p2)])
        return not any(line.intersects(obstacle) for obstacle in self.obstacles)

    def _compute_sampling_bounds(self):
        points = [self.start.point, self.goal.point]

        for obstacle in self.obstacles:
            coords = np.asarray(obstacle.exterior.coords, dtype=float)
            if len(coords) > 0:
                points.extend(coords[:, :2])

        pts = np.asarray(points, dtype=float)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        padding = max(self.step, 1.0)
        return mins - padding, maxs + padding

    # ---------------- PATH ----------------

    def extract_path(self):
        if self.goal.parent is None:
            return None

        path = []
        node = self.goal

        while node is not None:
            path.append(node.point)
            node = node.parent

        return path[::-1]
