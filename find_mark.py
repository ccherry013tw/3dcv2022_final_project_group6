import open3d as o3d
import cv2
import numpy as np
import sys

object_name = sys.argv[1]
def visualize(mesh,mark):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.add_geometry(mark)
    vis.run()
    vis.destroy_window()

def calcualte_mark(vertices):
    ref_points = np.loadtxt(f'models/{object_name}_mark.txt')
    f = open(f'models/{object_name}_pt.txt', 'w')
    for pt in ref_points:
        print(pt)
        diff = vertices - pt
        min_idx = np.argmin(np.linalg.norm(diff, axis = 1))
        f.write(f'{min_idx}\n')
        print(min_idx, vertices[min_idx])
    f.close()
    

def load_mark_point(vertices):
    mark = o3d.geometry.PointCloud()

    idx = np.loadtxt(f'models/{object_name}_pt.txt').astype(int)
    points = [vertices[i] for i in idx]

    mark.paint_uniform_color([1, 0, 0])
    mark.points = o3d.utility.Vector3dVector(points)

    return mark

def main():
    # mesh = o3d.io.read_triangle_mesh(f'models/{object_name}.obj', enable_post_processing=True)
    # img = o3d.io.read_image(f'models/{object_name}_texture.jpg')
    # mesh.textures = [o3d.geometry.Image(img)]
    # vertices = np.asarray(mesh.vertices).copy()
    mesh = o3d.io.read_point_cloud(f'models/{object_name}.pcd')
    vertices = np.asarray(mesh.points).copy() 
    print(vertices.shape)
    
    calcualte_mark(vertices)
    mark = load_mark_point(vertices)

    visualize(mesh, mark)

main()