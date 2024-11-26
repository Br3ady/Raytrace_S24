import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt


def circumcenter_and_radius(a, b, c, d):
    # make relative mat
    A = np.array([b - a, c - a, d - a])
    squared_lengths = np.array([np.dot(b - a, b - a), np.dot(c - a, c - a), np.dot(d - a, d - a)])
    
    circumcenter_rel_a = np.linalg.solve(2 * A.T, squared_lengths)
    circumcenter = circumcenter_rel_a + a
    
    # The squared radius of the circumcircle
    circumradius_squared = np.dot(circumcenter - a, circumcenter - a)
    
    return circumcenter, circumradius_squared

# Function to check if a point lies inside the circumsphere of a tetrahedron
def circumsphere_contains(a, b, c, d, point):
    circumcenter, circumradius_squared = circumcenter_and_radius(a, b, c, d)
    if circumcenter is None:
        return False  # Degenerate tetrahedron
    return np.dot(circumcenter - point, circumcenter - point) < circumradius_squared

# Delaunay triangulation function in 3D
def delaunay_triangulation_3d(points):
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    delta_max = np.max(max_coords - min_coords)
    mid_coords = (min_coords + max_coords) / 2
    
    # Super tetrahedron vertices
    p1 = tuple(mid_coords + np.array([3 * delta_max, 0, -delta_max]))
    p2 = tuple(mid_coords + np.array([-delta_max, 3 * delta_max, delta_max]))
    p3 = tuple(mid_coords + np.array([-delta_max, -3 * delta_max, delta_max]))
    p4 = tuple(mid_coords + np.array([0, 0, 3 * delta_max]))
    
    # Initial tetrahedron list
    tetrahedrons = [(p1, p2, p3, p4)]
    
    # Add points one at a time
    for point in points:
        point = tuple(point)
        bad_tetrahedrons = []
        faces = []
        
        # Step 1: Identify bad tetrahedrons whose circumspheres contain the point
        for tetra in tetrahedrons:
            if circumsphere_contains(np.array(tetra[0]), np.array(tetra[1]), np.array(tetra[2]), np.array(tetra[3]), np.array(point)):
                bad_tetrahedrons.append(tetra)
        
        # Step 2: Find the boundary faces of the polygonal hole
        print(bad_tetrahedrons)
        for tetra in bad_tetrahedrons:
            for face in [(tetra[0], tetra[1], tetra[2]), (tetra[0], tetra[1], tetra[3]), 
                         (tetra[0], tetra[2], tetra[3]), (tetra[1], tetra[2], tetra[3])]:
                # Sort the face tuples to ensure that comparisons are order-independent
                face = tuple(sorted(face))
                if face not in faces:
                    faces.append(face)
                else:
                    faces.remove(face)
        
        # Step 3: Remove bad tetrahedrons
        tetrahedrons = [t for t in tetrahedrons] #if t not in bad_tetrahedrons]
        
        # Step 4: Re-triangulate the polygonal hole
        for face in faces:
            tetrahedrons.append((face[0], face[1], face[2], point))
    
    # Step 5: Remove tetrahedrons connected to the super tetrahedron
    tetrahedrons = [t for t in tetrahedrons if p1 not in t and p2 not in t and p3 not in t and p4 not in t]
    
    return tetrahedrons


def generate_sphere_points(num_points):
    phi = np.random.uniform(0, 2 * np.pi, num_points)  
    theta = np.arccos(1 - 2 * np.random.uniform(0, 1, num_points))  

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.vstack((x, y, z)).T

# Example usage with a set of 3D points
points = generate_sphere_points(10)  # Generate 10 random points in 3D

# Perform Delaunay triangulation
tetrahedrons = delaunay_triangulation_3d(points)

# Visualize the result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the tetrahedrons
for tetra in tetrahedrons:
    verts = [list(tetra)]
    poly = Poly3DCollection(verts, alpha=0.3, edgecolor='k')
    ax.add_collection3d(poly)

ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r')  # Plot points
plt.title("3D Delaunay Triangulation")
plt.show()
