import argparse
import open3d as o3d
import numpy as np
import os

def pcd_to_npy(pcd_file):
    # Check if the input file exists
    if not os.path.isfile(pcd_file):
        print(f"Error: File '{pcd_file}' not found.")
        return

    # Read PCD file
    pcd = o3d.io.read_point_cloud(pcd_file)

    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)
    # Generate output file name
    output_file = os.path.splitext(pcd_file)[0] + ".npy"
    # Save numpy array as .npy file
    np.save(output_file, points)

    print(f"Point cloud saved as {output_file}")

    try:
        normals = np.asarray(pcd.normals)
        normals_output_file = os.path.splitext(pcd_file)[0] + "_normals.npy"
        np.save(normals_output_file, normals)
        print(f"Point Normals data saved as {normals_output_file}")
    except:
        print("No normals in cloud data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PCD file to NPZ file")
    parser.add_argument('-p','--pcd_file', help="Path to the input PCD file", required=True)
    args = vars(parser.parse_args())

    pcd_to_npy(args['pcd_file'])


    # pcd_to_npy("HAP_sweep_ds.pcd")
