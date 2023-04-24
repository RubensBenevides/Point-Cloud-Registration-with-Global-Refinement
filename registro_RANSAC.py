import open3d as o3d
import numpy as np 
import minhas_funcoes as myf
import time


def registration_RANSAC(source, target, voxel_size):
    # Cria o descritor FPFH
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=20))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=20))
    kd_tree_fpfh = o3d.geometry.KDTreeSearchParamHybrid(radius=10*voxel_size, max_nn=200)
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source, kd_tree_fpfh)
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target, kd_tree_fpfh)
    # Configuração do algoritmo de registro
    distance_threshold = voxel_size*2  # recomended = 2*voxel_down_sampling
    estimacao = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    checagem = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(0.3)]
    iterations = int((len(source.points) + len(target.points))/2)
    criterios_parada = o3d.pipelines.registration.RANSACConvergenceCriteria(iterations*5, 1.0)
    min_n_points = 3
    print("Realizando Registro RANSAC based...")
    # Realiza o registro global
    registration_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, 
        target, 
        source_fpfh,
        target_fpfh, 
        True,
        distance_threshold,
        estimacao,
        min_n_points,
        checagem,
        criterios_parada)
    return registration_result

# One parameter to rule them all
voxel_size = 0.1

source = o3d.io.read_point_cloud(f"nuvens/nuvens_pre_processadas/Arch/s{2}.pcd")
print(source)
target = o3d.io.read_point_cloud(f"nuvens/nuvens_pre_processadas/Arch/s{1}.pcd")
print(target)

ini = time.time()
T = registration_RANSAC(source, target, voxel_size)
print(f"Time taken: {np.round(time.time() - ini)}")
print(T)
myf.desenhar_resultado_registro(source, target, T.transformation)