import open3d as o3d
import numpy as np
import time
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


# SCRIPT TO DO COARSE PAIRWISE REGISTRATION USING FAST GLOBAL REGISTRATION
# ON NCLT DATASET


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


# Fast Global Registration (FGR) adapted to register a pair of TLS point clouds.
# INPUT: (source,target) -> a pair of point cloud. (voxel_size) the voxels used
# to downsample the original point cloud.
# OUTPUT: Open3D object containing RMSE between correspondences, fitness (sobreposition),
# list of correspondences and the 4x4 relative transformation. 
def registro_FGR(source, target, voxel_size):
    n_pontos = int((len(source.points) + len(target.points))/2)
    # Estimate normals
    kd_tree_normais = o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=20)
    source.estimate_normals(kd_tree_normais)
    target.estimate_normals(kd_tree_normais)
    # Calculate FPFH features
    kd_tree_descritores = o3d.geometry.KDTreeSearchParamHybrid(radius=10*voxel_size, max_nn=200)
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source,kd_tree_descritores)
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target,kd_tree_descritores)
    # Define FGR parameters
    FGR_coarse = o3d.pipelines.registration.FastGlobalRegistrationOption(
        division_factor = 1.4,       # padrao: 1.4
        use_absolute_scale = False,   # padrao: False (False is better)
        decrease_mu = True,          # padrao: False
        maximum_correspondence_distance = 2*voxel_size, # used to end the decrease_mu
        iteration_number = 300,      # padrao: 64
        tuple_scale      = 0.95,     # padrao: 0.95
        maximum_tuple_count = int(n_pontos*0.2)) # padrao: 1000
    # Apply FGR
    result_FGR = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(source,
                                                                                       target,
                                                                                       source_fpfh,
                                                                                       target_fpfh,
                                                                                       FGR_coarse)
    return result_FGR


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


# LOAD DATASET CLOUDS
n_clouds = 901
clouds = [o3d.io.read_point_cloud(f"nuvens/nuvens_pre_processadas/NCLT/s{i}.pcd") for i in range(n_clouds)]


# FGR REGISTRATION LOOP FOR CIRCUIT PAIRS
voxel_size = 0.1
results_FGR = []
times_FGR = []
for i in range(n_clouds):
    if i < n_clouds-1:
        print(f"Registering cloud {i+1} in cloud {i}")
        #ini = time.time()
        result_FGR = registro_FGR(clouds[i+1], clouds[i], voxel_size)
        #time_FGR = time.time()-ini
        #print(f"Pair {i}->{i+1} time taken: {round(time_FGR,2)} sec")
        #print(f"Pair {i}->{i+1} RMSE: {round(result_FGR.inlier_rmse,2)}")
        #times_FGR.append(time_FGR)
        results_FGR.append(result_FGR)
    elif i == n_clouds-1:
        print(f"Registering cloud {0} in cloud {i}")
        result_FGR = registro_FGR(clouds[0], clouds[i], voxel_size)
        results_FGR.append(result_FGR)


# RECOVER ABSOLUTES POSES FROM RELATIVE POSES
relative_poses_FGR = [results_FGR[i].transformation for i in range(n_clouds)]
absolute_poses_FGR = poses_relativas_para_absolutas(relative_poses_FGR)


# DRAW RESULT
apply_poses_in_clouds(absolute_poses_FGR, clouds)


# IMPORT GROUNDTRUTH AND SUBTRACT POSES
groundtruth = [np.loadtxt(f"groundtruth/NCLT/pose{i}.txt") for i in range(n_clouds)]
distances_R, distances_t = subtract_squared_poses(absolute_poses_FGR, groundtruth)


# PLOT POSE ERROR (DIFERENCES IN TRANSLATION)
plt.plot(distances_t, label = "FGR")
# Name X and Y axis
plt.xlabel('Absolute poses')
plt.ylabel('Error (m)')
plt.grid(True)
plt.legend()
plt.show()


# SAVE TRANSFORMATIONS
for i in range(n_clouds):
    np.savetxt(f"relative_poses_FGR/NCLT/pose_{i+1}_{i}.txt", results_FGR[i].transformation, fmt="%.10f")
