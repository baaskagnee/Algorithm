import pytest
from app import bfs_fewest_hops, dfs_all_simple_paths, dijkstra
adj = {
    1: [(2, 1.0), (3, 2.0)],
    2: [(4, 1.0)],
    3: [(4, 5.0)],
    4: []
}
def test_bfs_found():
    path, hops = bfs_fewest_hops(adj, 1, 4)
    assert path in ([1, 2, 4], [1, 3, 4])   # 2 hop
    assert hops == len(path) - 1


def test_bfs_no_path():
    g = {1: [(2, 1.0)], 2: [], 3: []}
    path, hops = bfs_fewest_hops(g, 1, 3)
    assert path is None
    assert hops >= 1


def test_bfs_same_node():
    path, hops = bfs_fewest_hops(adj, 1, 1)
    assert path == [1]
    assert hops == 0

def test_dfs_simple():
    path, explored = dfs_all_simple_paths(adj, 1, 4)
    assert path in ([1, 2, 4], [1, 3, 4])
    assert explored >= 1


def test_dfs_no_path():
    g = {1: [(2, 1)], 2: [], 5: []}
    path, explored = dfs_all_simple_paths(g, 1, 5)
    assert path is None
    assert explored >= 1


def test_dfs_same_node():
    path, explored = dfs_all_simple_paths(adj, 1, 1)
    assert path == [1]
    assert explored == 1

def test_dijkstra_shortest():
    """Dijkstra жинтэй богино замыг зөв сонгох."""
    dist, path = dijkstra(adj, 1, 4)
    # Шалгах: 1 → 2 → 4 = 2 km
    assert path == [1, 2, 4]
    assert pytest.approx(dist) == 2.0


def test_dijkstra_no_path():
    g = {1: [(2, 1)], 2: [], 7: []}
    dist, path = dijkstra(g, 1, 7)
    assert path is None
    assert dist == float("inf")


def test_dijkstra_same_node():
    dist, path = dijkstra(adj, 2, 2)
    assert path == [2]
    assert dist == 0.0
