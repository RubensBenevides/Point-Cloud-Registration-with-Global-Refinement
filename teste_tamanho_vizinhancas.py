import numpy as np
import open3d as o3d
import minhas_funcoes as myf

# Load point cloud
pcd = o3d.io.read_point_cloud("nuvens/nuvens_pre_processadas/Office/s0.pcd")

# Paint and build kd-tree
pcd.paint_uniform_color([0.5, 0.5, 0.5])
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

# paint an anchor point
print("Paint the 1501st point red.")
pcd.colors[1500] = [1, 0, 0]

# Find it's 200 nearest neighbors and paint them blue
[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], 200)
np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]

# draw
o3d.visualization.draw_geometries([pcd])


# Find its neighbors with distance less than 0.75, and paint them green
[k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], .75)
np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]

# draw
o3d.visualization.draw_geometries([pcd])
