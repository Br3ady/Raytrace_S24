import numpy as np
import matplotlib.pyplot as plt


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def intersect_sphere(ray_origin, ray_dir, sphere):
    sphere_to_ray = ray_origin - sphere["center"] 
    a = np.dot(ray_dir, ray_dir)
    b = 2 * np.dot(ray_dir, sphere_to_ray)
    c = np.dot(sphere_to_ray, sphere_to_ray) - sphere["radius"] ** 2
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return None 
    
    t1 = (-b - np.sqrt(discriminant)) / (2 * a)
    t2 = (-b + np.sqrt(discriminant)) / (2 * a)
    
    if t1 > 0:
        return t1 
    if t2 > 0:
        return t2
    return None

def intersect_plane(ray_origin, ray_dir, plane):
    denom = np.dot(plane["normal"], ray_dir)
    if np.abs(denom) > 1e-6:
        t = np.dot(plane["point"] - ray_origin, plane["normal"]) / denom
        if t >= 0:
            return t
    return None
 
def compute_lighting(point, normal, scene, light_dir):
    intensity = 0
    for light in scene["lights"]:
        light_pos = light["position"]
        light_color = light["color"]
        to_light = normalize(light_pos - point)
        intensity += light["intensity"] * max(0, np.dot(normal, to_light))
    return intensity

def trace_ray(ray_origin, ray_dir, scene):
    closest_t = np.inf
    closest_object = None
    intersection_point = None
    normal = None
    
    ###TODO change to mesh and universal polygon intersection
    for obj in scene["objects"]:
        if obj["type"] == "sphere":
            t = intersect_sphere(ray_origin, ray_dir, obj)
        elif obj["type"] == "plane":
            t = intersect_plane(ray_origin, ray_dir, obj)
        else:
            t = None
        
        if t is not None and t < closest_t:
            closest_t = t
            closest_object = obj
            
            # via scaling ray by t
            intersection_point = ray_origin + t * ray_dir
            if obj["type"] == "sphere":
                normal = normalize(intersection_point - obj["center"]) # for inensity scacle
            elif obj["type"] == "plane":
                normal = obj["normal"]
    
    if closest_object is None:
        return np.array([0, 0, 0])
    
    intensity = compute_lighting(intersection_point, normal, scene, ray_dir)    
    color = closest_object["color"] * intensity
    
    return np.clip(color, 0, 1)

## Main
def render(scene, camera, image_width, image_height):
    aspect_ratio = image_width / image_height
    image = np.zeros((image_height, image_width, 3))  ## Image with RGB channels
    
    viewport_height = 2.0
    viewport_width = viewport_height * aspect_ratio
    focal_length = 2.0  # dist from camera to viewport

    origin = camera["position"]
    horizontal = np.array([viewport_width, 0, 0])
    vertical = np.array([0, viewport_height, 0])
    lower_left_corner = origin - horizontal/2 - vertical/2 - np.array([0, 0, focal_length])

    for y in range(image_height):
        for x in range(image_width):
            u = x / (image_width - 1) ## fraction of total width covered so far
            v = (image_height - y - 1) / (image_height - 1)

            ray_dir = normalize(lower_left_corner + u * horizontal + v * vertical - origin)

            color = trace_ray(origin, ray_dir, scene) 
            image[y, x] = color
    
    return image



if __name__ == "__main__":

    scene = {
        "objects": [
            {
                "type": "sphere",
                "center": np.array([0, 0, -5]),
                "radius": 1,
                "color": np.array([1, 0, 0])  
            },
            {
                "type": "sphere",
                "center": np.array([2, 0, -5]),
                "radius": 1,
                "color": np.array([0, 1, 0])  
            },
            {
                "type": "plane",
                "point": np.array([0, -1, 0]),
                "normal": np.array([0, 1, 0]),
                "color": np.array([0.5, 0.5, 0.5]) 
            }
        ],
        "lights": [
            {
                "type": "point",
                "position": np.array([5, 5, 5]),
                "color": np.array([1, 1, 1]),
                "intensity": 1.0
            }
        ]
    }
    camera = {
        "position": np.array([0, 0, 0]), 
    }


    image_width = 400
    image_height = 300
    image = render(scene, camera, image_width, image_height)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
