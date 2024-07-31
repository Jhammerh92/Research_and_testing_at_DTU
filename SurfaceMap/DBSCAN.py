import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def generate_sample_data(n_samples=500):
    """
    Generate sample 3D data points for clustering.
    """
    np.random.seed(0)
    centers = [[1, 1, 1], [-1, -1, -1], [1, -1, 1]]
    data = [np.random.normal(loc=center, scale=0.2, size=(n_samples // 3, 3)) for center in centers]
    outliers = np.random.uniform(-5, 5, size=(n_samples // 3, 3))
    data.append(outliers)
    data = np.vstack(data) 
    return data

def dbscan_clustering(data, eps=0.1, min_samples=3, standardize=True):
    """
    Perform DBSCAN clustering on 3D data points.

    Parameters:
    - data (numpy array): The 3D data points to be clustered.
    - eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - labels (numpy array): Cluster labels for each point in the dataset.
    """
    if standardize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    
    return labels

def plot_clusters(data, labels):
    """
    Plot the 3D clusters and outliers.

    Parameters:
    - data (numpy array): The 3D data points.
    - labels (numpy array): Cluster labels for each point.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'k'  # Black used for noise.
        
        class_member_mask = (labels == k)
        
        xyz = data[class_member_mask]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=col, edgecolor='k', s=50)
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title('DBSCAN Clustering')
    plt.show()

if __name__ == "__main__":
    # Generate sample data
    data = generate_sample_data()
    
    # Perform DBSCAN clustering
    labels = dbscan_clustering(data)
    
    # Plot the clusters
    plot_clusters(data, labels)
