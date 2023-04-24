import numpy as np
np.set_printoptions(suppress=True)
import open3d as o3d
import minhas_funcoes as myf




# Fine registration with Mulscale GICP. INPUTS: pair of clouds (source,target);
# 'current_voxel_size' = initital scale for multiscale registration; 'ini_T' = initial transformation.
# OUTPUT: object containing the result of the registration + Information Matrix 
def registrar_com_ICP_generalizado_multiescala(source, target, current_voxel_size, ini_T):
    voxel_radius = [5*current_voxel_size, 2.5*current_voxel_size, current_voxel_size, (1/2)*current_voxel_size, (1/4)*current_voxel_size]
    max_correspondence_distance = [4*voxel_radius[0], 3*voxel_radius[1], 2.5*voxel_radius[2], 2*voxel_radius[3], 2*voxel_radius[4]]
    # Criteria
    iteracoes = [500, 400, 300, 100, 100]
    fitness_var = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
    RMSE_var = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
    current_transformation = ini_T
    # Robust function for weigth correspondences
    for i in range(5):
        source = source.voxel_down_sample(voxel_radius[i])
        target = target.voxel_down_sample(voxel_radius[i])
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_radius[i]*2.5, max_nn=30))
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_radius[i]*2.5, max_nn=30))
        result_icp = o3d.pipelines.registration.registration_generalized_icp(source,
            target,
            max_correspondence_distance[i],
            current_transformation,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=fitness_var[i],
                relative_rmse=RMSE_var[i],
                max_iteration=iteracoes[i]))
        current_transformation = result_icp.transformation
    # Information Matrix
    information_matrix_ICP = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source,
        target,
        max_correspondence_distance[i],
        current_transformation)
    return result_icp, information_matrix_ICP


# Load clouds
n_nuvens = 13
nuvens = [o3d.io.read_point_cloud(f"C:/Users/ruben/Documents/DISSERTACAO/Datasets_originais/Originais/Bremen/s{i}.pcd") for i in range(n_nuvens)]
# Load initial poses
relative_poses = [np.loadtxt(f"relative_poses_FGR_GICP/Bremen/manuais_refinadas/pose_{i+1}_{i}.txt") for i in range(n_nuvens-1)]
loop_closure = np.loadtxt(f"relative_poses_FGR_GICP/Bremen/manuais_refinadas/pose_0_12.txt")
relative_poses.append(loop_closure)


# Refine all pairs
voxel_size = 0.1
results_GICP = []
initial_Ts = results_FGR
# Multiscale-GICP loop
for i in range(n_nuvens-1):
    print(f"Registering cloud {i+1} in cloud {i} with M-GICP")
    result_GICP, _ = registrar_com_ICP_generalizado_multiescala(nuvens[i+1], 
        nuvens[i], 
        voxel_size, 
        relative_poses[i])
    print(f"Fitness: {result_GICP.fitness}")
    print(f"RMSE: {result_GICP.inlier_rmse}")
    results_GICP.append(result_GICP)


# Evaluate RMSE/Fitness with 0.1 m distance
for i in range(n_nuvens-1):
    T = o3d.pipelines.registration.evaluate_registration(nuvens[i+1], 
        nuvens[i], 
        voxel_size*3, 
        results_GICP[i].transformation)
    print(f"Fitness: {T.fitness}")
    print(f"RMSE: {T.inlier_rmse}\n")


# Draw pairs
for i in range(n_nuvens-1):
    myf.desenhar_resultado_registro(nuvens[i+1], nuvens[i], results_GICP[i].transformation) 


# Save transformation
for i in range(n_nuvens-1):
    np.savetxt(f"relative_poses_FGR_GICP/Bremen/manuais_refinadas/pose_{i+1}_{i}.txt", results_GICP[i].transformation)

