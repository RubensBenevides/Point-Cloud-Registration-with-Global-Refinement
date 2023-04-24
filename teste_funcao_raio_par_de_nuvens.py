import numpy as np 
import open3d as o3d

# TODO: executar depeois que acabar o registro do dataset Arch

def sphere_radi(source,target):
    xyz_max_1 = source.get_max_bound()
    xyz_min_1 = source.get_min_bound()
    xyz_max_2 = target.get_max_bound()
    xyz_min_2 = target.get_min_bound()
    box_diagonal_1 = sum((xyz_max_1 - xyz_min_1)**2)**(0.5)
    box_diagonal_2 = sum((xyz_max_2 - xyz_min_2)**2)**(0.5)
    mean_radi = (box_diagonal_1*box_diagonal_2*0.25)**(0.5)
    return mean_radi

# Load 2 clouds
pc_1 = o3d.io.read_point_cloud("nuvens/nuvens_pre_processadas/Office/s0.pcd")
pc_2 = o3d.io.read_point_cloud("nuvens/nuvens_pre_processadas/Office/s4.pcd")

# Calculate mean radios of enclosing sphere
radi = sphere_radi(pc_1,pc_2)

# Create sphere mesh and sample points on surface
esfera = o3d.geometry.TriangleMesh.create_sphere(radius=radi,resolution=20)
esfera = esfera.sample_points_uniformly(number_of_points=10000)

esfera_1 = o3d.geometry.TriangleMesh.create_sphere(radius=radi/1.2,resolution=20)
esfera_1 = esfera_1.sample_points_uniformly(number_of_points=10000)

esfera_2 = o3d.geometry.TriangleMesh.create_sphere(radius=radi/(1.2*1.2),resolution=20)
esfera_2 = esfera_2.sample_points_uniformly(number_of_points=10000)

esfera_3 = o3d.geometry.TriangleMesh.create_sphere(radius=radi/(1.2*1.2*1.2),resolution=20)
esfera_3 = esfera_3.sample_points_uniformly(number_of_points=10000)

# Draw
o3d.visualization.draw_geometries([pc_1,pc_2,esfera,esfera_1,esfera_2,esfera_3])
