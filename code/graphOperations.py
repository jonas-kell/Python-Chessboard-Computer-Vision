import numpy as np

nodes_count = 0
nodes = []
weights_matrix = None

# corner_candidates: [(x,y)]
# entry: (x,y)
def filter_by_graph_method(corner_candidates, entry, number_of_nodes_to_filter):
    global nodes_count
    global nodes

    nodes_count = len(corner_candidates) + 1

    nodes = []
    nodes.append(entry)
    for cor in corner_candidates:
        nodes.append(cor)

    init_graph()

    allowed_dist = 64
    while True:
        contained_nodes = bfs(0, allowed_dist, [])

        # not =, because 0 index is start(center)
        if len(contained_nodes) > number_of_nodes_to_filter:

            result = []
            for i in range(number_of_nodes_to_filter):
                result.append(corner_candidates[contained_nodes[i + 1] - 1])
            return result

        allowed_dist *= 2


def init_graph():
    global nodes_count
    global nodes
    global weights_matrix

    weights_matrix = np.zeros((nodes_count, nodes_count), dtype=np.uint32)
    for i in range(nodes_count):
        for j in range(i + 1, nodes_count):
            dist = (nodes[i][0] - nodes[j][0]) ** 2 + (nodes[i][1] - nodes[j][1]) ** 2

            weights_matrix[i, j] = dist
            weights_matrix[j, i] = dist


def bfs(start, max_trav_dist, visited=[]):
    global nodes_count
    global weights_matrix

    if start not in visited:
        visited.append(start)

    for neighbour in range(nodes_count):
        if (weights_matrix)[start, neighbour] < max_trav_dist:
            if neighbour not in visited:
                bfs(start=neighbour, max_trav_dist=max_trav_dist, visited=visited)

    return visited


if __name__ == "__main__":
    test_nodes = [(10, 10), (40, 40), (20, 20), (0, 10)]
    print(filter_by_graph_method(test_nodes, (0, 0), 3))

    test_nodes = [(40, 40), (10, 10), (20, 20), (0, 10)]
    print(filter_by_graph_method(test_nodes, (0, 0), 3))

    test_nodes = [(0, 40000), (0, 1000), (0, 2000), (0, 3000)]
    print(filter_by_graph_method(test_nodes, (0, 0), 3))
