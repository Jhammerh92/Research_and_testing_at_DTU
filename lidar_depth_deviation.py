# import os
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from LivoxSynthesizer import load_synthetic


FOLDER = 'livox40_synthetic_10s3m_0'
points, std_profile_radial =  load_synthetic(FOLDER)

# induced error
# t = np.linspace(np.deg2rad(40),0,100)
# amp =  np.sin(t*15)**2* (1-np.exp(-3*t)) *0.02 + 0.01

# polar coordinates
thetas = np.arctan2(np.linalg.norm(points[:,0:2], axis=1), points[:,2])
rhos = np.arctan2(points[:,1], points[:,0])+ np.pi
depths = np.linalg.norm(points,axis=1)


radial_edge_max = np.max(abs(thetas))*1.01
azimuth_edge_max = 2*np.pi

n = 25
# radial_bin_edges = np.linspace(0, radial_edge_max, n+1, endpoint=True)
radial_bin_edges = np.sin((np.linspace(0.0, np.arcsin(radial_edge_max), n+1, endpoint=True)))
azimuth_bin_edges = np.linspace(0, azimuth_edge_max, n+1,endpoint=True)


# print(len(bins_edges))
# bin_length = 2/n*edge_max
# bin_area = bin_length**2
print(radial_edge_max)

# print(bins_edges)

radial_bins = np.digitize(thetas, radial_bin_edges) - 1
azimuth_bins = np.digitize(rhos, azimuth_bin_edges) - 1
bucket_index = np.c_[radial_bins, azimuth_bins]
linear_bucket_index = [y*n + x for x,y in bucket_index]
# unique_buckets = np.unique(linear_bucket_index)
buckets = [[] for _ in range(n**2)]
_ = [buckets[b].append(i) for i,b in enumerate(linear_bucket_index)]

# average_distance = np.array([depths[idx] for idx in buckets])
average_distance = np.array([np.mean(depths[idx]*np.cos(thetas[idx])) for idx in buckets],dtype='float64').reshape((n,n))
average_distance = np.where(average_distance>0, average_distance, np.nan)


std_distance = np.array([np.std(depths[idx]*np.cos(thetas[idx])) for idx in buckets]).reshape((n,n))
std_distance = np.where(std_distance>0, std_distance, np.nan)
counts = np.array([len(idx) for idx in buckets]).reshape((n,n))
counts = np.where(counts>0, counts, np.nan)

mean_std_distance = np.mean(std_distance, axis=0) 


print()

radial_bin_centers = (radial_bin_edges[1:] + radial_bin_edges[:-1])/2
azimuth_bin_centers = (azimuth_bin_edges[1:] + azimuth_bin_edges[:-1])/2
radial_mesh, azimuth_mesh = np.meshgrid(radial_bin_centers, azimuth_bin_centers)

mean_radial_distance = np.mean(radial_mesh, axis=0) 

azimuth_steps = azimuth_bin_edges[1:]-azimuth_bin_edges[:-1]
radial_steps = radial_bin_edges[1:]**2-radial_bin_edges[:-1]**2

radial_step_mesh, azimuth_step_mesh = np.meshgrid(radial_steps, azimuth_steps)

areas = radial_step_mesh * np.pi * azimuth_step_mesh

# counts, _, _ = np.histogram2d(points[:,0], points[:,1], bins=[bins, bins])
count_density = counts/areas *1e-4# counts/bin_area *1e-4 

# fig, axes = plt.subplots(1, 3)
fig = plt.figure()

for i,(data, title) in enumerate(zip([count_density, average_distance, std_distance],
                        ['Point density', 'Average Projected Depth', 'Standard Deviation'])):
    ax = fig.add_subplot(130+i+1,projection='polar')
    im = ax.pcolormesh(azimuth_mesh, np.rad2deg(radial_mesh), data, cmap='Reds')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='50%', pad=0.05, )
    fig.colorbar(im, orientation='vertical')
    ax.set_title(title)




# im1 = axes[0].imshow(count_density)
# divider = make_axes_locatable(axes[0])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# axes[1].imshow(average_distance)
# axes[2].imshow(std_distance, cmap='Reds')

# fig.colorbar(im1, cax=cax, orientation='vertical')



# radial_std = 

fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
ax.plot_wireframe(np.rad2deg(azimuth_mesh), np.rad2deg(radial_mesh),
                   std_distance, rstride=1, cstride=1)
ax.set_xlabel(r'Azimuth Angle  $\rho \, [{}^\circ]$')
ax.set_ylabel(r'Angle from center  $\theta \, [{}^\circ]$')
ax.set_zlabel('Std $\sigma, \, [m]$')
# ax.set_title("Column (x) stride set to 0")

fig = plt.figure()
plt.scatter(np.rad2deg(azimuth_mesh.ravel()), std_distance.ravel(), alpha=0.1)
plt.ylabel('Std $\sigma, \, [m]$')
plt.xlabel(r'Azimuth Angle  $\rho \, [{}^\circ]$')

fig = plt.figure()
# # plt.plot(average_distance[n//2, n//2:])
# # plt.plot(np.arange(n//2+2)-1, std_distance[n//2, n//2-1:])
# # plt.plot(count[n//2, n//2:])
# plt.scatter(radial_mesh.ravel(), std_distance.ravel()/np.cos(np.arcsin(radial_mesh.ravel()/3)), alpha=0.1)
plt.scatter(np.rad2deg(radial_mesh.ravel()), std_distance.ravel(), alpha=0.1)


# plt.plot(t, amp*np.cos(t), color='C1', lw=3)
# plt.plot(np.rad2deg(t), amp, color='C1', lw=3)
plt.plot(np.rad2deg(std_profile_radial[:,0]), std_profile_radial[:,1], color='C1', lw=3)
plt.plot(np.rad2deg(mean_radial_distance), mean_std_distance, color='C3',ls='--', lw=3)

plt.title('Radial Std')
plt.ylabel('Std $\sigma, \, [m]$')
plt.xlabel(r'Angle from center  $\theta \, [{}^\circ]$')





# fig = plt.figure()
# plt.plot(np.tan(t)*3, amp, color='C1', lw=3)
# plt.scatter(std_dist_from_center[:,1], std_dist_from_center[:,0]/np.cos(np.arcsin(std_dist_from_center[:,1]/3)), alpha=0.1)


plt.show()
