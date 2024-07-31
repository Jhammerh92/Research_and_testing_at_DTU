import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import distance

from plane_analysis_functions import *


class SurfaceQuadtreeNode:
    def __init__(self, center, size, depth=0, parent=None, normal=None, quadrant=0, tree=None):
        self.center = np.array(center)
        self.size = size
        self.depth = depth
        self.has_points = False
        self.points = []
        self.normals = []
        self.indices = []
        # self.children = [None] * 4
        self.children = [None]
        self.parent = parent
        self.normal = normal
        self.curvature = 0.0
        self.normal_point = self.center
        self.normal_distance = 0.0
        self.is_plane = False
        self.basis = None
        self.quadrant = quadrant
        self.init_corners(self.center)
        self.tree = tree
        if tree is None:
            self.tree = parent.tree

        self.height = 0.0

        self.precalc_quadrants = []


    def init_corners(self, center):
        half_size = self.size/2
        x = np.linspace(-half_size, half_size, 2) + self.center[0]
        y = np.linspace(-half_size, half_size, 2) + self.center[1]
        X,Y = np.meshgrid(x, y)
        Z = np.zeros_like(X) 
        self.corner_points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    def is_leaf(self):
        return all(child is None for child in self.children)
    
    def is_surface(self):
        return not self.normal is None
    
    def create_children(self, data):
        cx, cy = self.center
        half = self.size / 2
        quarter = self.size / 4
        children_centers = [
            (cx - quarter, cy - quarter),
            (cx - quarter, cy + quarter),
            (cx + quarter, cy - quarter),
            (cx + quarter, cy + quarter)
        ]

        depth = self.depth + 1

        # for i in range(4):
        #     child = SurfaceQuadtreeNode(children_centers[i], half, depth, parent=self, quadrant=i, tree=self.tree)
        #     self.children[i] = child
        self.children = [SurfaceQuadtreeNode(children_centers[i], half, depth, parent=self, quadrant=i) for i in range(4)]
        

        # for point, index in data:
        #     quadrant = self._get_quadrant_index(point)
        #     self.children[quadrant]._insert(point, index)

    def get_siblings(self):
        if self.parent is None: # in case self is root, then there is no parent
            return []
        all_siblings = self.parent.get_children()
        valid_siblings = [sibling for sibling in all_siblings if sibling.points and sibling != self ]
        return valid_siblings
    
    def get_children(self):
        children = []
        if self.is_leaf():
            return [self]
        else:
            for child in self.children:
                if child is not None:
                    children.extend(child.get_children())
            return children
        
        # if node.is_leaf():
        #     return [node]
        # else:
        #     leaves = []
        #     for child in node.children:
        #         if child is not None:
        #             leaves.extend(self._collect_leaf_nodes(child))
        #     return leaves


    def _insert(self, point, index, normals):
        self.points.append(point)
        self.indices.append(index)
        self.normals.append(normals)
        quadrant = self._get_quadrant_index(point)
        self.precalc_quadrants.append(quadrant)

    def _insert_bulk(self, points, indices, normals):
        self.points.extend(points)
        self.indices.extend(indices)
        self.normals.extend(normals)
        # precalculate quadrants vectorised
        quadrants = self.calc_quadrant_indices(points)
        self.precalc_quadrants.extend(quadrants)

    def calc_quadrant_indices(self, points):
        q_index = np.zeros(len(points), dtype=np.int8)
        np_points = np.array(points)
        q_index[np_points[:,0] > self.center[0]] |= 2
        q_index[np_points[:,1] > self.center[1]] |= 1
        return q_index


    def _get_quadrant_index(self, point):
        index = 0
        if point[0] > self.center[0]:
            index |= 2
        if point[1] > self.center[1]:
            index |= 1
        return index

    def get_precalc_quadrant_indices(self):
        return self.precalc_quadrants
    
    def get_plane_information(self):
        return self.normal, self.normal_point
    
    def plot_plane(self, ax, color=None):

        # Generate a grid of points in the XY plane
        # size = self.size/2
        # point_grid = np.linspace(-size, size, 2)
        # X, Y = np.meshgrid(point_grid, point_grid)
        # Z = np.zeros_like(X) 
        # X += center[0]
        # Y += center[1]
        # Z += origodist

        # Create the grid of points
        # grid_points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        grid_points = self.corner_points

        # Calculate the rotation matrix to align Z-axis with the normal
        # def rotation_matrix_from_vectors(vec1, vec2):
        #     """ Find the rotation matrix that aligns vec1 to vec2 """
        #     a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        #     v = np.cross(a, b)
        #     c = np.dot(a, b)
        #     s = np.linalg.norm(v)
        #     kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        #     rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        #     return rotation_matrix

        # R = rotation_matrix_from_vectors(np.array([0, 0, 1]), normal)

        # local_normal_basis = construct_orthonormal_basis(normal @ R_basis[:3,:3])

        # elevation_from_normal(normal @ R_basis[:3,:3], grid_points)

        # Rotate the grid points to align with the normal vector
        # rotated_grid_points = grid_points @ R.T
        R_basis = self.tree.basis
        rotated_grid_points =  homogenous_transformation(grid_points, R_basis, pre=True)

        # Offset the grid to the center position
        X_rot = rotated_grid_points[:, 0]
        Y_rot = rotated_grid_points[:, 1]
        Z_rot = rotated_grid_points[:, 2]

        # Reshape the points back to the grid shape
        X_rot = X_rot.reshape((2,2))
        Y_rot = Y_rot.reshape((2,2))
        Z_rot = Z_rot.reshape((2,2))

        # ls = LightSource(270, 45)
        # color = rotated_grid_points[:,2]
        # fc = ls.shade(color, cmap=plt.cm.gray, vert_exag=0.1, blend_mode='soft')

        # Plot the plane
        # fc = ls.shade_normals(normal)
        # fc = ls.shade(mad, cmap='jet')
        # ax.plot_surface(X_rot, Y_rot, Z_rot, alpha=0.5, color=color, lightsource=ls, linewidth=1, edgecolor='k')
        ax.plot_surface(X_rot, Y_rot, Z_rot, alpha=0.5, linewidth=1, edgecolor='k', color=color)
        # ax.plot_surface(X_rot, Y_rot, Z_rot, alpha=1.0, facecolor=fc, linewidth=1)


        return ax


        

    

class SurfaceQuadtree:
    def __init__(self, points, normals, indices, min_density=0.0, max_points=4):
        self.min_density = min_density
        # self.max_density = max_density
        self.max_points = max_points
        self.depth_nodes = [[]]  # List to keep track of nodes at each depth level
        self.points = points # the actual 3d points in the cloud
        self.normals = normals # the actual 3d points in the cloud
        self.indices = indices # the indices of the points so they be retrieved easily

        self.center = np.mean(points, axis=0)
        self.surface_normal = np.median(normals, axis=0)
        plane_basis = construct_orthonormal_basis(self.surface_normal).T # construct a orthonormal basis from the normal vector of the plane

        self.basis = np.eye(4)
        self.basis[:3, :3] = plane_basis
        self.R = plane_basis
        # if not basis is None:
        #     self.R = basis
        
        self.basis[:3,3] = self.center

        self.flattened_points = homogenous_transformation(points, np.linalg.inv(self.basis), pre=True)
        self.rotated_normals =  normals @ self.R
        _, size = self.calculate_bounds(self.flattened_points[:,:2])
        self.root = SurfaceQuadtreeNode(np.zeros(2), size, tree=self)
        self.depth_nodes[0].append(self.root)


        for point, index, normal in zip(self.flattened_points, indices, self.rotated_normals):
            self.insert(point, index, normal)
        
        self.refine_based_on_min_density(self.root)
        self.refine_based_on_child_count(self.root)
        # self.calculate_node_height(self.root)
        self.calculate_node_planes(self.root)
        # self._reassign_single_points()


    def calculate_bounds(self, points):
        points = np.array(points)
        center = np.mean(points, axis=0)
        max_dist = np.max(np.linalg.norm(points - center, axis=1))
        # max_dist = np.max(points - center)
        size = max_dist * 2
        return center, size
    
    def point_within_tree(self, point, plane_dist_limit=1.0):
        flattened_point = homogenous_transformation(point, np.linalg.inv(self.basis), pre=True).ravel()
        half_size = self.root.size / 2.0
        return np.abs(flattened_point[0]) < half_size and np.abs(flattened_point[1]) < half_size and np.abs(flattened_point[2]) < plane_dist_limit


    def insert(self, point, index, normal):
        self._insert(self.root, point, index, normal)

    def _insert(self, node:SurfaceQuadtreeNode, point, index, normal):
        if node.is_leaf():
            node._insert(point, index, normal)
            node.has_points = True # this cannot go False again since points will not be removed
            # area = node.size ** 2
            # density = len(node.points) / area
            # density_bool = len(node.points) > 5 and density >= self.max_density
            if len(node.points) > self.max_points:# or density_bool:
                self._subdivide(node)
        else:
            quadrant = self._get_quadrant_index(node, point)
            # quadrant = node.get_quadrant()
            self._insert(node.children[quadrant], point, index, normal)

    def _insert_to_quadrants(self, node:SurfaceQuadtreeNode, points, indices, quadrants):
        for quadrant in np.unique(quadrants):
            node.children[quadrant]._insert_bulk(points, indices)

    def get_node(self, point):
        flattened_point = homogenous_transformation(point, transformation=np.linalg.inv(self.basis), pre=True)
        return self._get_node(self.root, flattened_point)
        
    def _get_node(self, node:SurfaceQuadtreeNode, point):
        if node.is_leaf() and node.has_points: # and self.is_point_inside(node, point):
            return node
        elif not node.has_points: # if the first node has no points then no children will have points, therefore return here
            return None
        else:
            quadrant = self._get_quadrant_index(node, point)
            return self._get_node(node.children[quadrant], point)

    def is_point_inside(self, node:SurfaceQuadtreeNode, point):
        pass
        point = np.ravel(point)
        is_inside = True
        # if point[0] > node.center[0] + node.size:
        #     index |= 2
        # if point[1] > node.center[1]:
        #     index |= 1


    def _subdivide(self, node:SurfaceQuadtreeNode):
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

        node.children = [None]*4
        for i in range(4):
            child = SurfaceQuadtreeNode(children_centers[i], half, depth, parent=node)
            node.children[i] = child
            self.depth_nodes[depth].append(child)

        points = node.points
        indices = node.indices
        normals = node.normals
        node.points = []
        node.indices = []
        node.normals = []
        node.precalc_quadrants = []
        # quadrants = node.get_precalc_quadrant_indices()

        # node.create_children(zip(points, indices))
        # self._insert_to_quadrants(node, points, indices, quadrants)

        for point, index, normal in zip(points, indices, normals):
            self._insert(node, point, index, normal)


    def _get_quadrant_index(self, node, point):
        index = 0
        point = np.ravel(point)
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


    def find_neighbors(self, node:SurfaceQuadtreeNode):
        """Find all neighboring nodes of a given node."""
        neighbors = []
        if node.parent is not None:
            for sibling in node.get_siblings():
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

    def refine_based_on_min_density(self, node:SurfaceQuadtreeNode):
        if node.is_leaf() and node.points:
            area = node.size ** 3
            density = np.log(len(node.points)+1e-5) / area
            if density < self.min_density:
                self._subdivide(node)
                self.refine_based_on_min_density(node)
        # if not node.is_leaf():
        else:
            for child in node.children:
                if child is not None:
                    self.refine_based_on_min_density(child)

    def refine_based_on_child_count(self, node:SurfaceQuadtreeNode):
        if node.is_leaf() and len(node.points) > 1:
            if siblings := node.get_siblings():
                    sib_points = []
                    for sibling in siblings:
                        sib_points.extend(sibling.points)
                    quadrants = node.calc_quadrant_indices(sib_points)
                    n_quadrants = len(np.unique(quadrants))
                    if n_quadrants < 3:
                        self._subdivide(node)
                        self.refine_based_on_child_count(node)
        # if not node.is_leaf():
        else:
            for child in node.children:
                if child is not None:
                    self.refine_based_on_child_count(child)
        
    def calculate_node_height(self, node:SurfaceQuadtreeNode):
        if node.is_leaf() and node.points:
            height = np.median(node.points, axis=0)[2]
            node.height = height
            node.corner_points[:,2] =+ height
        # if not node.is_leaf():
        else:
            for child in node.children:
                if child is not None:
                    self.calculate_node_height(child)

    def calculate_node_planes(self, node:SurfaceQuadtreeNode):
        if node.is_leaf() and node.points:
            is_plane_bool, mp, mnv, mad, mad_angle_dist, origodist, curvature  = is_plane(node.points, node.normals, False)
            if not is_plane_bool:
                # collect points in siblings and check with those
                if siblings := node.get_siblings():
                    sib_points = []
                    sib_normals = []
                    for sibling in siblings:
                        sib_points.extend(sibling.points)
                        sib_normals.extend(sibling.normals)
                    
                    is_plane_bool = is_plane(sib_points, sib_normals)[0]

                # for sibling in node.get_siblings():
                # is_plane_bool = True # assume true when anded with siblings, if 1 is false then it becomes false again
                # for sibling in node.get_siblings():
                #     is_plane_bool &= is_plane(sibling.points, sibling.normals)[0]

            elevation_from_normal(mnv, node.corner_points, mp)

            node.height = mp[2] # not used!
            node.normal_point = homogenous_transformation(np.atleast_2d(mp), self.basis, pre=True)
            node.normal = mnv @ self.R.T
            node.normal_distance = plane_distance(node.normal_point, np.zeros(3), -node.normal)
            node.is_plane = is_plane_bool
            node.curvature = curvature[0]
            

        else:
            for child in node.children:
                if child is not None:
                    self.calculate_node_planes(child)

    def visualize(self):
        fig, ax = plt.subplots()
        colors = plt.cm.get_cmap('rainbow', 1024)
        ax.set_aspect('equal', 'box')

        def draw_node(node, depth=0):
            if node.is_leaf() and node.points:
                rand_c = np.random.randint(0, 1024)
                color = colors(rand_c)
                np_points = np.array(node.points)
                # for point in node.points:
                #     ax.plot(point[0], point[1], color=color)
                # ax.scatter(*np_points.T, color=color, s=3)
                ax.plot(*np_points[:,:2].T, '.', color=color)
                rect = Rectangle((node.center[0] - node.size / 2, node.center[1] - node.size / 2), 
                                 node.size, node.size, 
                                 fill=False, edgecolor=color, linewidth=1)
                ax.add_patch(rect)
            else:
                for child in node.children:
                    if child is not None:
                        draw_node(child, depth + 1)

        draw_node(self.root)
        # plt.show()
    
    def visualize_separate_planes(self, ax=None, with_points=False):
        if ax is None:
            fig = plt.figure(figsize=(8,6), layout='constrained')
            ax = fig.add_subplot(111, projection='3d')

        ax.set_aspect('equal', 'box')

        def draw_plane(node:SurfaceQuadtreeNode):
            if node.is_leaf() and node.points: # and node.is_plane:
                color = 'blue' if node.is_plane else 'red'
                node.plot_plane(ax, color=color)
                if with_points:
                    p = np.array(node.points)
                    p = homogenous_transformation(p, self.basis, pre=True)
                    ax.scatter(*p.T, s=10, alpha=0.5, c='k')
            else:
                for child in node.children:
                    if child is not None:
                        draw_plane(child)
        draw_plane(self.root)

        set_equal_aspect(ax)
        # plt.show()

    def visualize_continoues_plane(self):
        surface_points = []
        for leaf in self.leaf_nodes():
            surface_points.extend(leaf.corner_points)

        surface_points = homogenous_transformation(np.array(surface_points), self.basis, pre=True)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(*surface_points.T, antialiased=True, alpha=0.5)
        set_equal_aspect(ax)
        # plt.show()



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

    quadtree = SurfaceQuadtree(points, indices, min_density=0.1, max_points=2)
    quadtree.visualize()

    # Example of iterating over non-empty leaf nodes
    for leaf in quadtree.leaf_nodes():
        print(f'Leaf node at center {leaf.center} with points {leaf.points} and indices {leaf.indices}')
