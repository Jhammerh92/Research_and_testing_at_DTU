import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import distance

class QuadtreeNode:
    def __init__(self, center, size, depth=0, parent=None):
        self.center = np.array(center)
        self.size = size
        self.depth = depth
        self.points = []
        self.indices = []
        self.children = [None] * 4
        self.parent = parent

    def is_leaf(self):
        return all(child is None for child in self.children)

class Quadtree:
    def __init__(self, points, indices, min_density, max_points=4):
        center, size = self.calculate_bounds(points)
        self.root = QuadtreeNode(center, size)
        self.min_density = min_density
        self.max_points = max_points
        self.depth_nodes = [[]]  # List to keep track of nodes at each depth level
        self.depth_nodes[0].append(self.root)

        for point, index in zip(points, indices):
            self.insert(point, index)
        
        self.refine_based_on_density(self.root)
        self._reassign_single_points()

    def calculate_bounds(self, points):
        points = np.array(points)
        center = np.mean(points, axis=0)
        max_dist = np.max(np.linalg.norm(points - center, axis=1))
        size = max_dist * 2
        return center, size

    def insert(self, point, index):
        self._insert(self.root, point, index)

    def _insert(self, node, point, index):
        if node.is_leaf():
            node.points.append(point)
            node.indices.append(index)
            if len(node.points) > self.max_points:
                self._subdivide(node)
        else:
            quadrant = self._get_quadrant_index(node, point)
            self._insert(node.children[quadrant], point, index)

    def _subdivide(self, node):
        cx, cy = node.center
        half = node.size / 2
        quarter = node.size / 4
        children_centers = [
            (cx - quarter, cy - quarter),
            (cx - quarter, cy + quarter),
            (cx + quarter, cy - quarter),
            (cx + quarter, cy + quarter)
        ]

        depth = node.depth + 1
        if depth >= len(self.depth_nodes):
            self.depth_nodes.append([])

        for i in range(4):
            child = QuadtreeNode(children_centers[i], half, depth, parent=node)
            node.children[i] = child
            self.depth_nodes[depth].append(child)

        points = node.points
        indices = node.indices
        node.points = []
        node.indices = []

        for point, index in zip(points, indices):
            self._insert(node, point, index)

    def _get_quadrant_index(self, node, point):
        index = 0
        if point[0] > node.center[0]:
            index |= 2
        if point[1] > node.center[1]:
            index |= 1
        return index

    def _reassign_single_points(self):
        """Reassign single points to their nearest neighbor node with more than one point."""
        for depth in reversed(range(len(self.depth_nodes))):
            for node in self.depth_nodes[depth]:
                if node.is_leaf() and len(node.points) == 1:
                    point, index = node.points[0], node.indices[0]
                    neighbors = self.find_neighbors(node)

                    valid_neighbors = [neighbor for neighbor in neighbors if len(neighbor.points) > 1]

                    if valid_neighbors:
                        nearest_neighbor = min(valid_neighbors, key=lambda n: distance.euclidean(point, n.center))
                        nearest_neighbor.points.append(point)
                        nearest_neighbor.indices.append(index)
                        node.points = []
                        node.indices = []

    def find_neighbors(self, node):
        """Find all neighboring nodes of a given node."""
        neighbors = []
        if node.parent is not None:
            for sibling in node.parent.children:
                if sibling is not None and sibling != node:
                    if sibling.is_leaf():
                        neighbors.append(sibling)
                    else:
                        neighbors.extend(self._collect_leaf_nodes(sibling))
            # Check for other potential neighbors at the same depth
            for other_node in self.depth_nodes[node.depth]:
                if other_node is not node and self._are_neighbors(node, other_node):
                    neighbors.append(other_node)
        return neighbors

    def _collect_leaf_nodes(self, node):
        """Recursively collect all leaf nodes."""
        if node.is_leaf():
            return [node]
        else:
            leaves = []
            for child in node.children:
                if child is not None:
                    leaves.extend(self._collect_leaf_nodes(child))
            return leaves

    def _are_neighbors(self, node1, node2):
        """Check if two nodes are neighbors."""
        dist = distance.euclidean(node1.center, node2.center)
        max_dist = (node1.size + node2.size) / 2
        return dist <= max_dist

    def refine_based_on_density(self, node):
        if node.is_leaf() and node.points:
            area = node.size ** 2
            density = len(node.points) / area
            if density < self.min_density:
                self._subdivide(node)
                self.refine_based_on_density(node)
        # if not node.is_leaf():
        else:
            for child in node.children:
                if child is not None:
                    self.refine_based_on_density(child)

    def visualize(self):
        fig, ax = plt.subplots()
        colors = plt.cm.get_cmap('rainbow', 1024)
        ax.set_aspect('equal', 'box')

        def draw_node(node, depth=0):
            if node.is_leaf() and node.points:
                rand_c = np.random.randint(0, 1024)
                color = colors(rand_c)
                for point in node.points:
                    ax.plot(point[0], point[1], 'o', color=color)
                rect = Rectangle((node.center[0] - node.size / 2, node.center[1] - node.size / 2), 
                                 node.size, node.size, 
                                 fill=False, edgecolor=color, linewidth=1)
                ax.add_patch(rect)
            else:
                for child in node.children:
                    if child is not None:
                        draw_node(child, depth + 1)

        draw_node(self.root)
        plt.show()

    def leaf_nodes(self):
        """Generator to iterate over all non-empty leaf nodes in the quadtree."""
        def _leaf_nodes(node):
            if node.is_leaf():
                if node.points:
                    yield node
            else:
                for child in node.children:
                    if child is not None:
                        yield from _leaf_nodes(child)

        yield from _leaf_nodes(self.root)

if __name__ == "__main__":
    points = [
        (0.1, 0.1),
        (-0.5, -0.5),
        (0.5, 0.5),
        (0.9, 0.9),
        (-0.9, -0.9),
        (0.3, -0.7),
        (-0.3, 0.7),
        (0.6, -0.2),
        (-0.6, 0.2),
    ]

    indices = list(range(len(points)))

    quadtree = Quadtree(points, indices, min_density=0.1, max_points=2)
    quadtree.visualize()

    # Example of iterating over non-empty leaf nodes
    for leaf in quadtree.leaf_nodes():
        print(f'Leaf node at center {leaf.center} with points {leaf.points} and indices {leaf.indices}')
