import numpy as np
np.set_printoptions(suppress=True)
import copy
import open3d as o3d
import minhas_funcoes as myf



# Funcao que seleciona as variaveis necessarias para executar os resultados de um dataset. 
# Nao reproduz o resultado, pois demoraria demais, apenas carrega o resultado salvo.
def switch(dataset):
    if dataset == "1":
        dataset = "Facade"
        print("You'll see Facade results.")
        path = "nuvens/nuvens_pre_processadas/Facade/"
        n_nuvens = 7
        paths = [path + f"s{i}.pcd" for i in range(n_nuvens)]
        return n_nuvens, paths, dataset
    elif dataset == "2":
        dataset = 'Courtyard'
        print("You'll see Courtyard results.")
        path = "nuvens/nuvens_pre_processadas/Courtyard/"
        n_nuvens = 8
        paths = [path + f"s{i}.pcd" for i in range(n_nuvens)]
        return n_nuvens, paths, dataset
    elif dataset == "3":
        dataset = "Arch"
        print("You'll see Arch results.")
        path = "nuvens/nuvens_pre_processadas/Arch/"
        n_nuvens = 5
        paths = [path + f"s{i}.pcd" for i in range(n_nuvens)]
        return n_nuvens, paths, dataset
    elif dataset == "4":
        dataset = "Bremen"
        print("You'll see Bremen results.")
        path = "nuvens/nuvens_pre_processadas/Bremen/"
        n_nuvens = 13
        paths = [path + f"s{i}.pcd" for i in range(n_nuvens)]
        return n_nuvens, paths, dataset
    elif dataset == "5":
        dataset = "NCLT"
        print("You'll see NCLT results.")
        path = "nuvens/nuvens_pre_processadas/NCLT/"
        n_nuvens = 901
        paths = [path + f"s{i}.pcd" for i in range(n_nuvens)]
        return n_nuvens, paths, dataset
    elif dataset == "6":
        dataset = "Office"
        print("You'll see Office results.")
        path = "nuvens/nuvens_pre_processadas/Office/"
        n_nuvens = 5
        paths = [path + f"s{i}.pcd" for i in range(n_nuvens)]
        return n_nuvens, paths, dataset


# Distance to analize RMSE and sobreposition
voxel_size = 0.1


# Obtain the dataset variables acording user input
dataset = input("Select the dataset:\n 1 = Facade\n 2 = Courtyard\n 3 = Arch\n 4 = Bremen\n 5 = NCLT (MLS)\n 6 = Office\n")
n_nuvens, paths, dataset = switch(dataset)


# Load pre-processed clouds
nuvens = [o3d.io.read_point_cloud(paths[i]) for i in range(n_nuvens)]


# Load relative poses
if dataset == "Bremen" or dataset == "Arch":
    relative_poses = [np.loadtxt(f"relative_poses_FGR_GICP/{dataset}/manuais_refinadas/pose_{i+1}_{i}.txt") for i in range(n_nuvens-1)]
    loopclosure = np.loadtxt(f"relative_poses_FGR_GICP/{dataset}/manuais_refinadas/pose_0_{n_nuvens-1}.txt")
    relative_poses.append(loopclosure)
elif dataset == "Facade" or dataset == "Courtyard" or dataset == "NCLT" or dataset == "Office":
    relative_poses = [np.loadtxt(f"relative_poses_FGR_GICP/{dataset}/pose_{i+1}_{i}.txt") for i in range(n_nuvens-1)]
    loopclosure = np.loadtxt(f"relative_poses_FGR_GICP/{dataset}/pose_0_{n_nuvens-1}.txt")
    relative_poses.append(loopclosure)


# Compose relative poses to absolute poses
absolute_poses = myf.poses_relativas_para_absolutas(relative_poses)


# Show the dataset with poses before optimizations
print("Clouds before optimizations")
myf.apply_poses_in_clouds(absolute_poses, nuvens)


# The LUM part of the SLEEP+LUM optimization can be personalized with weights.
# We use fitness (sobreposition) to improve results.
print("Calculating RMSE and fitness of pairs...")
RMSE_before_MRG, fitness_before_MRG = myf.calculate_RMSE_and_fitness(nuvens, relative_poses, voxel_size)
weights = list(np.ones(n_nuvens))
weights = fitness_before_MRG


# Calculate optimized absolute poses with SLERP+LUM (last pose = closure error)
absolute_poses_SLERP_LUM = myf.reconstruir_Ts_para_origem_SLERP_LUM(relative_poses, weights)
# Tnsert idendity as first pose
absolute_poses_SLERP_LUM.insert(0,np.identity(4))
del absolute_poses_SLERP_LUM[-1]


# Apply the fully optimized absolute poses in the clouds
print("Clouds after SLERP+LUM optimization")
myf.apply_poses_in_clouds(absolute_poses_SLERP_LUM, nuvens)


# Plot differences between absolute poses before/after the MRG SLERP+LUM-3D
poses_groundtruth = [np.loadtxt(f"groundtruth/{dataset}/pose{i}.txt") for i in range(n_nuvens)]
dif_R_before, dif_t_before = myf.subtract_poses(poses_groundtruth, absolute_poses)
dif_R_after,  dif_t_after  = myf.subtract_poses(poses_groundtruth, absolute_poses_SLERP_LUM)


# Plot
myf.plot_antes_depois_translacoes(dif_t_before, dif_t_after)
myf.plot_antes_depois_rotacoes(dif_R_before, dif_R_after)


# Percentage of total error relative to circuit length
print(f"Percentage of reduction in rotations: {1-(np.mean(dif_R_after)/np.mean(dif_R_before))}")
print(f"Percentage of reduction in translations: {1-(np.mean(dif_t_after)/np.mean(dif_t_before))}")

# TODO: Adicionar dataset Trees