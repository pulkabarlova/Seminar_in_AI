"""
PRM planner adapted from the PythonRobotics ProbabilisticRoadMap example:
https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/ProbabilisticRoadMap/probabilistic_road_map.py
"""

import random
import math
import networkx as nx
from shapely.geometry import LineString, Point
from scipy.spatial import KDTree


class PRM:
    def __init__(self, obstacles,
                 rand_area=(-1, 10),
                 n_samples=200,
                 k=10,
                 max_edge_len=None):

        self.obstacles = obstacles
        self.min_rand, self.max_rand = rand_area
        self.n_samples = n_samples
        self.k = k
        self.max_edge_len = max_edge_len

        self.samples = []
        self.graph = nx.Graph()
        self.roadmap_samples = []
        self.roadmap_graph = nx.Graph()
        self._roadmap_kdtree = None

    def plan(self, start, goal):
        self.samples = self._sample_points(include_start_goal=(start, goal))
        self.graph = nx.Graph()

        self._build_graph(self.samples, self.graph)

        path = nx.shortest_path(self.graph,
                                source=start,
                                target=goal,
                                weight='weight')
        return path

    def build_roadmap(self):
        self.roadmap_samples = self._sample_points()
        self.roadmap_graph = nx.Graph()
        self._roadmap_kdtree = self._build_graph(self.roadmap_samples, self.roadmap_graph)

        return self.roadmap_graph

    def query(self, start, goal):
        if self._roadmap_kdtree is None or self.roadmap_graph.number_of_nodes() == 0:
            self.build_roadmap()

        if not self.is_free(start) or not self.is_free(goal):
            raise nx.NetworkXNoPath("start_or_goal_inside_obstacle")

        if start == goal:
            return [start]

        added_nodes = []
        added_edges = []

        try:
            for point in (start, goal):
                if point not in self.roadmap_graph:
                    self.roadmap_graph.add_node(point)
                    added_nodes.append(point)

                for neighbor in self._query_neighbors(self._roadmap_kdtree, self.roadmap_samples, point, self.k):
                    if point == neighbor:
                        continue
                    if self.roadmap_graph.has_edge(point, neighbor):
                        continue
                    if self.is_collision_free(point, neighbor):
                        self.roadmap_graph.add_edge(point, neighbor, weight=self.distance(point, neighbor))
                        added_edges.append((point, neighbor))

            return nx.shortest_path(
                self.roadmap_graph,
                source=start,
                target=goal,
                weight='weight',
            )
        finally:
            self.roadmap_graph.remove_edges_from(added_edges)
            self.roadmap_graph.remove_nodes_from(added_nodes)

    def get_neighbors(self, p):
        if not self.samples:
            return []
        sample_kdtree = KDTree(self.samples)
        return self._query_neighbors(sample_kdtree, self.samples, p, self.k)

    def is_free(self, p):
        point = Point(p)
        return not any(obstacle.contains(point) for obstacle in self.obstacles)

    def is_collision_free(self, p1, p2):
        if self.max_edge_len is not None and self.distance(p1, p2) >= self.max_edge_len:
            return False

        line = LineString([p1, p2])
        return not any(line.intersects(obstacle) for obstacle in self.obstacles)

    def distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _sample_points(self, include_start_goal=None):
        samples = []

        if include_start_goal is not None:
            start, goal = include_start_goal
            samples.extend([start, goal])

        while len(samples) < self.n_samples:
            point = (
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand),
            )
            if self.is_free(point):
                samples.append(point)

        return samples

    def _build_graph(self, samples, graph):
        if not samples:
            return None

        sample_kdtree = KDTree(samples)

        for point in samples:
            graph.add_node(point)

        for point in samples:
            for neighbor in self._query_neighbors(sample_kdtree, samples, point, self.k + 1):
                if neighbor == point:
                    continue
                if not graph.has_edge(point, neighbor) and self.is_collision_free(point, neighbor):
                    graph.add_edge(point, neighbor, weight=self.distance(point, neighbor))

        return sample_kdtree

    @staticmethod
    def _query_neighbors(kdtree, samples, point, k):
        k = min(k, len(samples))
        if k <= 0:
            return []

        _, indexes = kdtree.query(point, k=k)
        if k == 1:
            indexes = [int(indexes)]

        return [samples[int(index)] for index in indexes]
