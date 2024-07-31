import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import open3d as o3d
from LivoxSynthesizer import load_synthetic
from LivoxScanParser import LivoxScanParser
from PlotPointCloud3d import plot_cloud_3d_o3d_path, get_points_from_pcd, set_axes_equal



# t = np.linspace(np.deg2rad(40),0,100)
# amp =  np.sin(t*15)**2* (1-np.exp(-3*t)) *0.02 + 0.01

def detect_plane_model(points):
    """
    docstring
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    plane_model, _ = point_cloud.segment_plane(distance_threshold=0.1,
                                         ransac_n=3,
                                         num_iterations=100)
    # extracted_plane = point_cloud.select_by_index(inliers)
    print((plane_model))
    return plane_model

def calc_normal_distances_to_plane(plane_model, points):
    """
    docstring
    """
    A,B,C,D = plane_model # Ax + by + cz + d
    distances = np.array([abs(A*x + B*y + C*z + D) / np.sqrt(A**2 + B**2 + C**2) for x,y,z in points])
    # print(distances)
    return distances

def plane_benchmarking_cartesian(points, n_grid, std_profile=None, edge_max=None):
    """
    docstring
    """
    if std_profile is None:
        std_profile = np.empty((1,1))


    x=0
    y=1
    z=2

    # n = n_grid
    edge_max_arr = np.empty([2,2])
    if edge_max is None:
        # edge_max_x = np.max(abs(np.linalg.norm(points[:,[y]],axis=1)))*1.01 # one percent beyond the max value
        edge_max_arr[0,0] = np.max(points[:,[y]])*1.01 # one percent beyond the max value
        edge_max_arr[0,1] = np.min(points[:,[y]])*1.01 # one percent beyond the max value
        # edge_max_y = np.max(abs(np.linalg.norm(points[:,[z]],axis=1)))*1.01 # one percent beyond the max value
        edge_max_arr[1,0] = np.max(points[:,[z]])*1.01 # one percent beyond the max value
        edge_max_arr[1,1] = np.min(points[:,[z]])*1.01 # one percent beyond the max value
        edge_max_index = np.argmax(abs(np.diff(edge_max_arr,axis=1)))
        edge_max_r = np.max(abs(edge_max_arr))
    else:
        edge_max_ = edge_max
        edge_max_ = edge_max

    # bins_edges = np.linspace(-edge_max_, edge_max_, n_grid+1, endpoint=True)
    bins_edges = [[],[]]
    bins_edges[edge_max_index] = np.linspace(edge_max_arr[edge_max_index,1], edge_max_arr[edge_max_index,0], n_grid+1, endpoint=True)
    bin_length = bins_edges[edge_max_index][1] - bins_edges[edge_max_index][0]
    bins_edges[edge_max_index^1] = np.arange(edge_max_arr[edge_max_index^1,1], edge_max_arr[edge_max_index^1,0] + bin_length, bin_length)

    grid_shape = (len(bins_edges[1])-1, len(bins_edges[0])-1)

    

    bin_area = bin_length**2
    print(bin_length, bin_area)
    print(grid_shape)

    # print(bins_edges)


    x_bins = np.digitize(points[:,y], bins_edges[0]) - 1
    y_bins = np.digitize(points[:,z], bins_edges[1]) - 1
    bucket_index = np.c_[x_bins, y_bins]
    linear_bucket_index = [y*grid_shape[1] + x for x,y in bucket_index]
    # unique_buckets = np.unique(linear_bucket_index)
    buckets = [[] for _ in range(grid_shape[0] * grid_shape[1])]
    _ = [buckets[b].append(i) for i,b in enumerate(linear_bucket_index)]


    plane_model = detect_plane_model(points)
    normal_distances_to_plane = calc_normal_distances_to_plane(plane_model, points)

    average_distance = np.array([np.mean(points[idx,x])  if len(points[idx,x])>0 else np.nan for idx in buckets]).reshape(grid_shape)
    plane_distance = np.array([np.mean(normal_distances_to_plane[idx])  if len(points[idx,x])>0 else np.nan for idx in buckets]).reshape(grid_shape)
    # average_distance = np.where(average_distance>0, average_distance, np.nan)
    std_distance = np.array([np.std(points[idx,x]) if len(points[idx,x])>0 else np.nan for idx in buckets]).reshape(grid_shape)
    std_plane = np.array([np.sqrt(np.sum(normal_distances_to_plane[idx]**2)/(len(normal_distances_to_plane[idx])-1)) if len(points[idx,x])>0 else np.nan for idx in buckets]).reshape(grid_shape)
    # std_distance = np.where(std_distance>0, std_distance, np.nan)
    counts = np.array([len(idx) if len(idx)>0 else np.nan for idx in buckets]).reshape(grid_shape)
    # counts = np.where(counts>0, counts, np.nan)


    print()


    # count_density = np.empty((n,n))
    # average_distance = np.empty((n,n))
    # std_distance = np.empty((n,n))

    # i = 0
    # for y in range(n):
    #     for x in range(n):
    #         idx = np.where(np.all(buckets == [x,y], axis=1))[0]
    #         pnts = points[idx, :]
    #         average_distance[y,x] = np.nan if len(pnts)<=1 else np.mean(pnts[:,2])
    #         std_distance[y,x] =  np.nan if np.std(pnts[:,2])==0 else np.std(pnts[:,2])
    #         count_density[y,x] = np.nan if len(pnts)==0 else len(pnts)/(bin_area*1e4)
    #         i+=1
            
    #         print("\r",i)


    bin_centers_x = (bins_edges[0][1:]+bins_edges[0][:-1])/2
    bin_centers_y = (bins_edges[1][1:]+bins_edges[1][:-1])/2
    dist_mesh_X, dist_mesh_Y = np.meshgrid(bin_centers_x, bin_centers_y)
    dist_XY = np.stack((dist_mesh_X, dist_mesh_Y))
    dist_from_center = np.linalg.norm(dist_XY,axis=0)

    std_to_dist_from_center = np.c_[std_distance.ravel(), dist_from_center.ravel()]

    r_bins_edges = np.linspace(0, edge_max_r*np.sqrt(2), int((n_grid+1)*np.sqrt(2)), endpoint=True)
    r_bins = np.digitize(std_to_dist_from_center[:,1], r_bins_edges) - 1
    r_buckets = [[] for _ in range(int((n_grid+1)*np.sqrt(2))-1)]
    _ = [r_buckets[b].append(i) for i,b in enumerate(r_bins)]
    mean_std_distance = np.array([np.nanmean(std_to_dist_from_center[idx, 0]) if len(idx) > 0 else np.nan for idx in r_buckets ])
    # mean_std_plane = np.array([np.nanmean(normal_distances_to_plane[idx])for idx in r_buckets])

    r_bin_centers = (r_bins_edges[1:]+r_bins_edges[:-1])/2


    bin_area_cm = bin_area *1e4
    print(bin_area_cm)
    # counts, _, _ = np.histogram2d(points[:,0], points[:,1], bins=[bins, bins])
    # count_density = counts/(1)
    count_density = counts/(bin_area_cm)
    count_density_log = np.log10(count_density)
    

    
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, data, title, clabel in zip(axes, [average_distance , plane_distance, count_density_log,  (std_plane)],
                                    [ 'Average Depth (X-coordinate)','Plane Distance (Normal distance)', 'Log Point density','Standard Deviation'],
                                    [ '$[m]$', '$[m]$','$[Log(cm^-2)]$', '$[m]$']):
        # im = ax.imshow(data, cmap='Reds')
        # im = ax.pcolor(dist_mesh_X, dist_mesh_Y,data, cmap='Reds')
        im = ax.pcolor(dist_mesh_X, dist_mesh_Y, data, cmap='jet')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        # axes[1].imshow(average_distance)
        # axes[2].imshow(std_distance, cmap='Reds')
        fig.colorbar(im, cax=cax, orientation='vertical')
        cax.get_yaxis().labelpad = 0
        cax.set_ylabel(clabel, rotation=90)
        ax.set_aspect('equal')
        ax.set_title(title)

    # fig.tight_layout()



    fig = plt.figure()
    plt.scatter(std_to_dist_from_center[:,1], std_to_dist_from_center[:,0], alpha=0.05)
    # plt.plot(np.tan(std_profile[:,0])*3, std_profile[:,1], color='C1', lw=3)
    plt.plot(r_bin_centers, mean_std_distance, color='C3',ls='--', lw=3)
    # plt.plot(np.tan(t)*3, amp, color='C1', lw=3)
    plt.title('No Correction Projection')
    plt.ylabel('Std $\sigma, \, [m]$')
    plt.xlabel('Distance from center $[m]$')


    fig = plt.figure()
    plt.scatter(std_to_dist_from_center[:,1], std_to_dist_from_center[:,0], alpha=0.1)
    # plt.plot(np.tan(std_profile[:,0])*3, std_profile[:,1]*np.cos(std_profile[:,0]), color='C1', lw=3)
    plt.plot(r_bin_centers, mean_std_distance, color='C3',ls='--', lw=3)
    # plt.plot(np.tan(t)*3, amp*np.cos(t), color='C1', lw=3)
    plt.title('Projected Truth - Seeing $\sigma_d$')
    plt.ylabel('Std $\sigma, \, [m]$')
    plt.xlabel('Distance from center $[m]$')

    fig = plt.figure()
    plt.scatter(std_to_dist_from_center[:,1], std_to_dist_from_center[:,0]/np.cos(np.arctan(std_to_dist_from_center[:,1]/3)), alpha=0.1)
    # plt.plot(np.tan(std_profile[:,0])*3, std_profile[:,1], color='C1', lw=3)
    plt.plot(r_bin_centers, mean_std_distance/np.cos(np.arctan(r_bin_centers/3)), color='C3',ls='--', lw=3)
    # plt.plot(np.tan(t)*3, amp, color='C1', lw=3)
    plt.title(r'Projected Error - Seeing $\sigma = \sigma_d / \cos(\theta)$ ')
    plt.ylabel(r'Std $\sigma,  [m]$')
    plt.xlabel(r'Distance from center $[m]$')
    
    # fig = plt.figure()
    # plt.scatter(std_to_dist_from_center[:,1], std_to_dist_from_center[:,0]/np.cos(np.arctan(std_to_dist_from_center[:,1]/3)), alpha=0.1)
    # # plt.plot(np.tan(std_profile[:,0])*3, std_profile[:,1], color='C1', lw=3)
    # plt.plot(r_bin_centers, mean_std_plane/np.cos(np.arctan(r_bin_centers/3)), color='C3',ls='--', lw=3)
    # # plt.plot(np.tan(t)*3, amp, color='C1', lw=3)
    # plt.title(r'Projected Error from plane - Seeing $\sigma = \sigma_d / \cos(\theta)$ ')
    # plt.ylabel('Std $\sigma, \, [m]$')
    # plt.xlabel('Distance from center $[m]$')

    colors = std_plane
    norm = plt.Normalize(np.nanmin(colors), np.nanmax(colors))
    color_map = plt.cm.viridis  # Change the colormap as needed

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.bar3d(dist_mesh_X.flatten(), dist_mesh_Y.flatten(), 0,
                bin_length, bin_length, std_plane.flatten(), color=color_map(norm(std_plane.flatten())), zsort='max', edgecolor='black', linewidth=bin_length*0.1)
    
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_map), ax=ax, pad=0.1)
    cbar.set_label('Standard Deviation')
    set_axes_equal(ax, on_axes='xy')

if __name__ == "__main__":

    # folder = 'livox40_synthetic_10s3m_0'
    # points, std_profile =  load_synthetic(folder)

    # parser = LivoxScanParser()
    # parser.load_from_csv("Mid-70 Test Data/Livox70_3s50cm.csv")
    # parser.load_from_csv("Mid-70 Test Data/Livox70_3s150cm.csv")
    # parser.load_from_csv("Mid-70 Test Data/Livox70_3s200cm.csv")

    # PCD_PATH = r"/Users/jhh/Documents/test_data_velodyne_syntese/20220523_122236/pcds/000123_frame_131003.508169.pcd"
    # PCD_PATH = r"/Users/jhh/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/ARLI - LOAM and LiDAR research/Data/static_recorded_clouds/static_recording_20231003_142409/static_cloud_0001_10.pcd"
    
    # PCD_PATH = r"/Users/jhh/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/ARLI - LOAM and LiDAR research/Data/HAP wall merged clouds/HAP_wall_125cm_cleaned.pcd"
    PCD_PATH = r"/Users/jhh/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/ARLI - LOAM and LiDAR research/Data/HAP wall merged clouds/HAP_wall_100cm_cleaned.pcd"

    # points = parser.get_points()
    points = get_points_from_pcd(PCD_PATH)

    N_HORIZONTAL_CELLS = 51#301
    plane_benchmarking_cartesian(points, N_HORIZONTAL_CELLS)


    plot_cloud_3d_o3d_path(PCD_PATH)
    plt.show()
