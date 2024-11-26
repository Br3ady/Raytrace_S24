import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

# Function to create a voxel grid from a point cloud
def create_voxel_grid(point_cloud, grid_size, bounds):
    """
    Create a voxel grid from a point cloud.
    
    Args:
        point_cloud (np.ndarray): Nx3 array of point cloud data.
        grid_size (int): The size of the grid in each dimension.
        bounds (tuple): The bounding box for the grid as (min_coord, max_coord).
        
    Returns:
        np.ndarray: 3D grid with scalar values representing the occupancy of each voxel.
    """
    min_coord, max_coord = bounds
    voxel_grid = np.zeros((grid_size, grid_size, grid_size))

    # Scale the point cloud to fit the grid
    scaled_points = (point_cloud - min_coord) / (max_coord - min_coord) * (grid_size - 1)
    scaled_points = scaled_points.astype(int)
    
    # Assign occupancy to the grid
    for point in scaled_points:
        x, y, z = point
        voxel_grid[x, y, z] = 1  # Mark this voxel as occupied

    return voxel_grid

# Example point cloud data (you can replace this with your actual point cloud)
point_cloud = np.random.rand(500, 3)

# def grid 
grid_size = 50
min_coord = np.min(point_cloud, axis=0)
max_coord = np.max(point_cloud, axis=0)
bounds = (min_coord, max_coord)

voxel_grid = create_voxel_grid(point_cloud, grid_size, bounds)
vertices, faces, normals, values = marching_cubes(voxel_grid, level=0.5)
vertices = vertices / (grid_size - 1) * (max_coord - min_coord) + min_coord

# visualize 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the polygon collection for the mesh
mesh = Poly3DCollection(vertices[faces], alpha=0.7, edgecolor='k')
ax.add_collection3d(mesh)

# Set the limits of the axes
ax.set_xlim(min_coord[0], max_coord[0])
ax.set_ylim(min_coord[1], max_coord[1])
ax.set_zlim(min_coord[2], max_coord[2])

plt.title("Marching Cubes Surface Reconstruction")
plt.show()
