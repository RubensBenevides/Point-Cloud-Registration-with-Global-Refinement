import open3d as o3d
import quaternion as quat
import numpy as np
import time
import copy
from matplotlib import pyplot as plt


# Function to draw registration result and preserv the original cloud.
# Both temporary clouds will be painted with random colors.
def desenhar_resultado_registro(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0.85,0.65,0.05])
    target_temp.paint_uniform_color([0.60,0.10,0.85])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


# Function that, given n absolute poses, applies them to each cloud n in the list of n clouds
# The first pose on the list should be the identity. 
# The first cloud in the list will be the global origin. 
# INPUT: lista_poses = list of absolute poses; lista_nuvens = list of clouds.
# OUTPUT: Open the open3d window and draw results. 
def apply_poses_in_clouds(lista_poses,lista_nuvens):
    if len(lista_nuvens) != len(lista_nuvens):
        raise ValueError("Quantidade de poses absolutas deve ser igual a quantidade de nuvens")
    n_poses = len(lista_poses)
    # criar copias para manter nuvens originais no lugar
    copias = copy.deepcopy(lista_nuvens)
    # Desenhar nuvens transformadas com FoV = 0 graus
    vis = o3d.visualization.Visualizer()
    vis.create_window(width = 1920, height = 1080)
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=0.0)
    # Aplicar poses nas copias, perceba que 
    for i in range(n_poses):
        copias[i].transform(lista_poses[i])
        vis.add_geometry(copias[i])
    vis.run()
    vis.destroy_window()


# Function to compose relative poses to absolute poses
# INPUT: T_circuito = list of relative poses: T10, T21, T32, ... Tn_n-1
# OUTPUT: List of absolute poses: T10, T20, T30, ... Tn_0
def poses_relativas_para_absolutas(T_circuito):
    # obter lista de rotacoes para a origem por composicao multiplicativa
    lista_rotacoes_origem = []
    for i in range(len(T_circuito)):
        LoopClosure = np.identity(3)
        for j in range (len(T_circuito)-i-1,-1,-1):        
            LoopClosure = LoopClosure@T_circuito[j][:3,:3] # compoe j matrizes de rotacao
        lista_rotacoes_origem.append(LoopClosure) # lista de rotacoes compostas para origem
    lista_rotacoes_origem = list(reversed(lista_rotacoes_origem)) # inverte a lista, ultima eh a LoopClosure
    # obter as translacoes para a origem (t20, t30, t40, etc.)
    t_ini = T_circuito[0][0:3,3]
    lista_translacoes_origem = [t_ini]
    for i in range(len(T_circuito)-1):
        t_posterior = lista_rotacoes_origem[i]@T_circuito[i+1][0:3,3] + t_ini
        t_ini = t_posterior
        lista_translacoes_origem.append(t_posterior)
    # Juntar (rotacao + translacao) em T(4x4):
    Lista_de_Poses = []
    for i in range(len(T_circuito)):
        rot_trans = np.hstack((lista_rotacoes_origem[i],np.transpose([lista_translacoes_origem[i]])))
        Pose = np.vstack((rot_trans,np.array([0,0,0,1])))
        Lista_de_Poses.append(Pose)
    # Insert identity pose:
    Lista_de_Poses.insert(0,np.identity(4))
    # delete pose closure error
    del Lista_de_Poses[-1]
    return Lista_de_Poses


# This fuction read 2 list of poses and subtract them. Return two lists 
# of distances: euclidean distances for pairs of translations; 
# Frobenious distances for pairs of rotations.
# INPUT: 2 lists with n poses each. OUTPUT: 2 lists of n distances (rot + trans).   
def subtract_squared_poses(list_poses_1, list_poses_2):
    # Check length
    if len(list_poses_1) != len(list_poses_2):
        raise Exception("The list of poses should be the same size")
    # Initialize lists
    distances_R = []
    distances_t = []
    # Loop to subtract poses
    for i in range(len(list_poses_1)):
        d_poses = list_poses_1[i] - list_poses_2[i]
        d_squared = d_poses**2
        # We normalize rotation distance by 2*2**(1/2).
        d_R = (sum(sum(d_squared[:3,:3]))**(1/2))/2*2**(1/2)
        d_t = sum(d_squared[:3,3])**(1/2)
        distances_R.append(d_R)
        distances_t.append(d_t)
    return distances_R, distances_t


def create_scales(n_scales):
    voxel_radios = 0.1
    voxel_radios = [voxel_radios+(0.1*i) for i in range(n_scales)]
    voxel_radios.reverse()
    return voxel_radios


def max_correspondence_distances(scales):
    n_scales = len(scales)
    if n_scales == 3:
        max_correspondence_distances = [3*scales[0], 2*scales[1], scales[2]]
    elif n_scales == 4:
        max_correspondence_distances = [3*scales[0], 2.5*scales[1], 2*scales[2], scales[3]]
    elif n_scales == 5:
        max_correspondence_distances = [3*scales[0], 2.5*scales[1], 2*scales[2], 1.5*scales[3], scales[4]]
    return max_correspondence_distances


# Multiscale GICP
# INPUT: source, target = pair of point clouds to be registered 
# n_scales = how much scales to use in the registration.
# OUTPUT: registration_result = Open3D object with correspondence set, RMSE, 
# fitness and a 4x4 transformation.
def Multiscale_GICP(source, target, n_scales, itera_escala, T_ini):
    # Create voxel sizes for downsampling
    voxel_sizes = create_scales(n_scales)
    # Calculate max_correspondences_distances using radios form point cloud pair
    search_distances = max_correspondence_distances(voxel_sizes)
    # Initialize M-GICP parameters
    knn_filtro = 30
    std_filtro = 1.0
    current_transformation = T_ini
    # Pesagem das correspondencias 
    loss = o3d.pipelines.registration.L1Loss()
    # loop das escalas
    for i in range(n_scales):
        # Downsample
        print(f"Escala: {i+1}")
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        # Amostragem
        source_down = source_temp.voxel_down_sample(voxel_sizes[i])
        target_down = target_temp.voxel_down_sample(voxel_sizes[i])
        # Filtragem
        source_clean, _ = source_down.remove_statistical_outlier(knn_filtro, std_filtro)
        target_clean, _ = target_down.remove_statistical_outlier(knn_filtro, std_filtro)
        # Estimar normais
        source_clean.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
        target_clean.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
        # Aplicar GICP
        result_icp = o3d.pipelines.registration.registration_generalized_icp(source_clean,
            target_clean,
            search_distances[i],
            current_transformation,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(loss),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=itera_escala))
        current_transformation = result_icp.transformation
    return result_icp


# 1 - LOAD NCLT POINT CLOUDS
n_clouds = 901
clouds = [o3d.io.read_point_cloud(f"nuvens/nuvens_pre_processadas/NCLT/s{i}.pcd") for i in range(n_clouds)]


# 2 - LOAD RELATIVE POSES (PREVIOUSLY ESTIMATED BY FGR)
relative_poses_FGR = [np.loadtxt(f"relative_poses_FGR/NCLT/pose_{i+1}_{i}.txt") for i in range(n_clouds-1)]
loop_closure = np.loadtxt(f"relative_poses_FGR/NCLT/pose_0_900.txt")
relative_poses_FGR.append(loop_closure)

'''
# 3 - APPLY MULTISCALE M-GICP REFINEMENT 
# Initialize lists
times_GICP = []
results_GICP = []
initial_T = relative_poses_FGR
# Define M-GICP parameters
iterations = 100
n_escales = 5 # Keep it between 5 and 3 scales
#  Run M-GICP LOOP FOR CIRCUIT PAIRS
for i in range(n_clouds):
    if (i < n_clouds-1):
        print(f"Registering cloud {i+1} in cloud {i} with M-GICP")
        ini = time.time()
        result_MGICP = Multiscale_GICP(clouds[i+1],
            clouds[i],
            n_escales, 
            iterations,
            initial_T[i])
        time_GICP = time.time()-ini
        print(f"Pair {i}->{i+1} time taken: {round(time_GICP,3)} sec")
        print(f"Pair {i}->{i+1} RMSE: {round(result_MGICP.inlier_rmse,3)} m\n")
        times_GICP.append(time_GICP)
        results_GICP.append(result_MGICP)
    # Loop closure case
    elif i == n_clouds-1:
        print(f"Registering cloud {0} in cloud {i} with M-GICP")
        ini = time.time()
        result_MGICP = Multiscale_GICP(clouds[0],
            clouds[i],
            n_escales, 
            iterations,
            initial_T[i])
        time_GICP = time.time()-ini
        print(f"Pair {0}->{i} time taken: {round(time_GICP,3)} sec")
        print(f"Pair {0}->{i} RMSE: {round(result_MGICP.inlier_rmse,3)} m\n")
        times_GICP.append(time_GICP)
        results_GICP.append(result_MGICP)


# 4 - RECOVER RELATIVE POSES AND SAVE
relative_poses_FGR_GICP = [results_GICP[i].transformation for i in range(n_clouds)]


# 5 - TRANSFORM RELATIVE POSES TO ABSOLUTE POSES
absolute_poses_FGR_GICP = poses_relativas_para_absolutas(relative_poses_FGR_GICP)
'''


# USE THIS TO NOT REPRODUCE ALL REGISTRATIONS AGAIN
# LOAD ALREADY SAVED M-GICP REFINED POSES
# Relative poses
relative_poses_FGR_GICP = [np.loadtxt(f"relative_poses_FGR_GICP/NCLT/pose_{i+1}_{i}.txt") for i in range(n_clouds-1)]
loop_closure = np.loadtxt(f"relative_poses_FGR_GICP/NCLT/pose_0_900.txt")
relative_poses_FGR_GICP.append(loop_closure)
# Absolute poses
absolute_poses_FGR_GICP = [np.loadtxt(f"absolute_poses_FGR_GICP/NCLT/pose{i}.txt") for i in range(n_clouds)]


# 6 - DRAW CIRCUIT
apply_poses_in_clouds(absolute_poses_FGR_GICP, clouds)


# 7 - LOAD GROUNDTRUTH
groundtruth = [np.loadtxt(f"groundtruth/NCLT/pose{i}.txt") for i in range(n_clouds)]


# 8 - SUBTRACT POSES
absolute_poses_FGR = poses_relativas_para_absolutas(relative_poses_FGR)
distances_R_1, distances_t_1 = subtract_squared_poses(absolute_poses_FGR, groundtruth)
distances_R_1, distances_t_2 = subtract_squared_poses(absolute_poses_FGR_GICP, groundtruth)


# 9 - PLOT POSE ERROR (DIFERENCES IN TRANSLATION)
plt.plot(distances_t_1, label = "FGR")
plt.plot(distances_t_2, label = "M-GICP")
# Name X and Y axis
plt.xlabel('Absolute poses')
plt.ylabel('Error (m)')
plt.grid(True)
plt.legend()
plt.show()


# SAVE RELATIVE AND ABSOLUTE POSES REFINED BY M-GICP
#for i in range(n_clouds):
#    np.savetxt(f"relative_poses_FGR_GICP/NCLT/pose_{i+1}_{i}.txt", relative_poses_FGR_GICP[i])
#    np.savetxt(f"absolute_poses_FGR_GICP/NCLT/pose_{i+1}_{i}.txt", absolute_poses_FGR_GICP[i])
