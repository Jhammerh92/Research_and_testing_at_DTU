import numpy as np
import matplotlib.pyplot as plt
from LivoxScanParser import PCD, LivoxScanParser
import open3d as o3d
import pathlib
# import matplotlib

def set_axes_equal(ax: plt.Axes, on_axes='xyz'):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """

    def _set_axes_radius(ax, origin, radius, on_axes=None):
        x, y, z = origin
        if 'x' in on_axes:
            ax.set_xlim3d([x - radius, x + radius])
        if 'y' in on_axes:
            ax.set_ylim3d([y - radius, y + radius])
        if 'z' in on_axes:
            ax.set_zlim3d([z - radius, z + radius])

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius, on_axes)


def plot_cloud_3d(points):
    """
    docstring
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    # ax.scatter(points[:,0], points[:,1],points[:,2], alpha= 1.0, s=2, c=points[:,2], cmap='bwr')
    ax.scatter(points[:,0], points[:,1],points[:,2], alpha= 0.5, s=2)
    # ax.scatter(points2[:,0], points2[:,1],points2[:,2], alpha= 1.0, s=2)
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)


def get_pcd_from_file(path):
    """
    docstring
    """
    suffix_ = pathlib.Path(path).suffix

    if suffix_ == ".csv":
        parser_ = PCD()
        parser_.load_from_csv(path)
        # plot_cloud_3d(points=parser.get_points()[:2500,:])
        points = parser_.get_points()
        # create open3d pcd from points
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(points)
    elif suffix_ == ".pcd":
        o3d_pcd = o3d.io.read_point_cloud(path)

    return o3d_pcd

def get_points_from_pcd(path):
    """
    docstring
    """
    pcd = o3d.io.read_point_cloud(path)
    points = np.array(pcd.points)
    return points

def _create_pcd_from_points(points, normals=None):
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points)
    if not (normals is None):
        o3d_pcd.normals = o3d.utility.Vector3dVector(normals)
    return o3d_pcd

def plot_cloud_3d_o3d_path(path:str):
    """
    docstring
    """
    pcd = get_pcd_from_file(path)
    o3d.visualization.draw_geometries([pcd])

def save_o3d_pcd(filename, points, normals=None, colors=None):
    if not isinstance(points, o3d.geometry.PointCloud ):
        pcd = _create_pcd_from_points(points, normals=normals)
    else:
        pcd = points
    if isinstance(colors, np.ndarray):
        norm = plt.Normalize(np.nanmin(colors), np.nanmax(colors))
        color_map = plt.cm.viridis  # Change the colormap as needed
        c= color_map(norm(colors.flatten()))
        pcd.colors = o3d.utility.Vector3dVector(c[:,:3])
    elif not (colors is None):
        pcd.paint_uniform_color(np.array(colors))

    o3d.io.write_point_cloud(filename,  pcd)

def plot_cloud_3d_o3d(points, normals=None, colors=None):
    """
    docstring
    """
    if not isinstance(points, o3d.geometry.PointCloud ):
        pcd = _create_pcd_from_points(points, normals=normals)
    else:
        pcd = points
    if isinstance(colors, np.ndarray):
        norm = plt.Normalize(np.nanmin(colors), np.nanmax(colors))
        color_map = plt.cm.viridis  # Change the colormap as needed
        c= color_map(norm(colors.flatten()))
        pcd.colors = o3d.utility.Vector3dVector(c[:,:3])
    elif not (colors is None):
        pcd.paint_uniform_color(np.array(colors))
        
    o3d.visualization.draw_geometries([pcd])



if __name__ == "__main__":

    # load from a csv - saved from Livox viewer
    # parser = LivoxScanParser()
    # parser.load_from_csv("Mid-70 Test Data/Livox70_3s200cm.csv")
    # plot_cloud_3d(points=parser.get_points()[:2500,:])


    # PCD_PATH = r"/Users/jhh/Documents/test_data_velodyne_syntese/20220523_122236/pcds/000123_frame_131003.508169.pcd"
    PCD_PATH = r"/Users/jhh/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/ARLI - LOAM and LiDAR research/Data/static_recorded_clouds/static_recording_20230929_144304/static_cloud_0000_0.pcd"

    # load from PCD saved from other sources such as from ros2 or velodyne parser
    # pcd = o3d.io.read_point_cloud(r"/Users/jhh/Documents/test_data_velodyne_syntese/20220523_122236/pcds/000123_frame_131003.508169.pcd")
    # o3d.visualization.draw_geometries([pcd])
    #                               zoom=0.3412,
    #                               front=[0.4257, -0.2125, -0.8795],
    #                               lookat=[2.6172, 2.0475, 1.532],
    #                               up=[-0.0694, -0.9768, 0.2024])

    # plot_cloud_3d_o3d("Mid-70 Test Data/Livox70_3s200cm.csv")

    plot_cloud_3d_o3d_path(PCD_PATH)
    print(get_points_from_pcd(PCD_PATH))

    plt.show()
