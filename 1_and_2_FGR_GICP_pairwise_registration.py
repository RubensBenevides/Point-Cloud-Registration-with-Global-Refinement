import open3d as o3d
import minhas_funcoes as myf
import numpy as np
import time
import matplotlib as p


# One parameter to rule them all
voxel_size = 0.1 # meters


# Funcao que seleciona as variaveis necessarias para executar os resultados de um dataset. 
# Nao reproduz o resultado, pois demoraria demais, apenas carrega o resultado salvo.
def switch(dataset):
    if dataset == "1":
        dataset = "Facade"
        print("You'll see Facade results: RMSE and Fitness from 7 pairwise registrations.")
        path = "nuvens/nuvens_pre_processadas/Facade/"
        n_nuvens = 7
        paths = [path + f"s{i}.pcd" for i in range(n_nuvens)]
        return n_nuvens, paths, dataset
    elif dataset == "2":
        dataset = "Courtyard"
        print("You'll see Courtyard results: RMSE and Fitness from 8 pairwise registrations.")
        path = "nuvens/nuvens_pre_processadas/Courtyard/"
        n_nuvens = 8
        paths = [path + f"s{i}.pcd" for i in range(n_nuvens)]
        return n_nuvens, paths, dataset
    elif dataset == "3":
        dataset = "Arch"
        print("You'll see Arch results: RMSE and Fitness from 5 pairwise registrations.")
        path = "nuvens/nuvens_pre_processadas/Arch/"
        n_nuvens = 5
        paths = [path + f"s{i}.pcd" for i in range(n_nuvens)]
        return n_nuvens, paths, dataset
    elif dataset == "4":
        dataset = "Bremen"
        print("You'll see Bremen results: RMSE and Fitness from 13 pairwise registrations.")
        path = "nuvens/nuvens_pre_processadas/Bremen/"
        n_nuvens = 13
        paths = [path + f"s{i}.pcd" for i in range(n_nuvens)]
        return n_nuvens, paths, dataset
    elif dataset == "5":
        dataset = "NCLT"
        print("You'll see NCLT results: RMSE and Fitness from 901 pairwise registrations.")
        path = "nuvens/nuvens_pre_processadas/NCLT/"
        n_nuvens = 901
        paths = [path + f"s{i}.pcd" for i in range(n_nuvens)]
        return n_nuvens, paths, dataset
    elif dataset == "6":
        dataset = "Office"
        print("You'll see Office results: RMSE and Fitness from 5 pairwise registrations.")
        path = "nuvens/nuvens_pre_processadas/Office/"
        n_nuvens = 5
        paths = [path + f"s{i}.pcd" for i in range(n_nuvens)]
        return n_nuvens, paths, dataset


# Define the dataset variables
dataset = input("Select the dataset:\n 1 = Facade\n 2 = Courtyard\n 3 = Arch\n 4 = Bremen\n 5 = NCLT (MLS)\n 6 = Office\n")
print("Do you want to reproduce the results or just load them?")
decision = input("Digit 1 to show/load results or 0 to reproduce them: ")


# Load dataset variables
n_nuvens, paths, dataset = switch(dataset)


# Load clouds
nuvens = [o3d.io.read_point_cloud(paths[i]) for i in range(n_nuvens)]


# Print estimated time to reproduce results
if decision == '0' and dataset == 'Courtyard':
    print("Estimated time to reproduce Courtyard results: 2300 seconds.")
elif decision == '0' and dataset == 'Facade':
    print("Estimated time to reproduce Facade results: 400 seconds.")
elif decision == '0' and dataset == 'Arch':
    print("Estimated time to reproduce Arch results: 39772 seconds.")
elif decision == '0' and dataset == 'Bremen':
    print("Estimated time to reproduce Bremen results: 86000 seconds.")
elif decision == '0' and dataset == 'NCLT':
    print("Estimated time to reproduce NCLT results: 4800 seconds.")
elif decision == '0' and dataset == 'Office':
    print("Estimated time to reproduce Office results: 130 seconds.")


if decision == '1' and (dataset == 'Facade' or dataset == 'Courtyard' or dataset == 'NCLT' or dataset == 'Office'):
    # READ SAVED RESULTS (FGR)
    relative_poses_FGR = [np.loadtxt(f"relative_poses_FGR/{dataset}/pose_{i+1}_{i}.txt") for i in range(n_nuvens-1)]
    loopclosure = np.loadtxt(f"relative_poses_FGR/{dataset}/pose_0_{n_nuvens-1}.txt")
    relative_poses_FGR.append(loopclosure)
    # READ SAVED RESULTS (Multiscale-GICP)
    relative_poses_FGR_GICP = [np.loadtxt(f"relative_poses_FGR_GICP/{dataset}/pose_{i+1}_{i}.txt") for i in range(n_nuvens-1)]
    loopclosure = np.loadtxt(f"relative_poses_FGR_GICP/{dataset}/pose_0_{n_nuvens-1}.txt")
    relative_poses_FGR_GICP.append(loopclosure)
    # Calculate pairwise RMSE
    RMSE_FGR, fitness_FGR = myf.calculate_RMSE_and_fitness(nuvens, relative_poses_FGR, voxel_size)
    RMSE_FGR_GICP, fitness_FGR_GICP = myf.calculate_RMSE_and_fitness(nuvens, relative_poses_FGR_GICP, voxel_size)
    # Calculate pairwise FITNESS
    if dataset == 'NCLT':
        myf.plot_RMSE_line(RMSE_FGR,RMSE_FGR_GICP)
        myf.plot_fitness_line(fitness_FGR,fitness_FGR_GICP)
    myf.plot_RMSE_BAR(RMSE_FGR,RMSE_FGR_GICP)
    myf.plot_Fitness_BAR(fitness_FGR,fitness_FGR_GICP)
    # Compose relative poses to obtain absolute poses
    absolute_poses_FGR = myf.poses_relativas_para_absolutas(relative_poses_FGR)
    absolute_poses_FGR_GICP = myf.poses_relativas_para_absolutas(relative_poses_FGR_GICP)
    # Apply absolute poses to draw before and after (all pairs must be correctly registered) 
    print("Showing 3D reconstruction with FGR only")
    myf.apply_poses_in_clouds(absolute_poses_FGR, nuvens)
    print("Showing 3D reconstruction with FGR+GICP")
    myf.apply_poses_in_clouds(absolute_poses_FGR_GICP, nuvens)
    # Print the enhancement in percetage of the RMSE and Fitness for all poses
    print(f"Percentage of reduction in RMSE: {1-(np.mean(RMSE_FGR_GICP)/np.mean(RMSE_FGR))}")
    print(f"Percentage of increase in sobreposition: {1-(np.mean(fitness_FGR)/np.mean(fitness_FGR_GICP))}")
# FOR DATASETS BREMEN AND ARCH, LOAD MANUAL POSES
elif decision == '1' and (dataset == 'Arch' or dataset == 'Bremen'):
    print(f"\nFGR was not able to register pairs of clouds from {dataset}.\nLoading manual and manual-refined relative poses to compare.")
    # READ PRE-SAVED RESULTS (FGR)
    relative_poses_FGR = [np.loadtxt(f"relative_poses_FGR/{dataset}/manuais/pose_{i+1}_{i}.txt") for i in range(n_nuvens-1)]
    loopclosure = np.loadtxt(f"relative_poses_FGR/{dataset}/manuais/pose_0_{n_nuvens-1}.txt")
    relative_poses_FGR.append(loopclosure)
    # READ PRE-SAVED RESULTS (Multiscale-GICP)
    relative_poses_FGR_GICP = [np.loadtxt(f"relative_poses_FGR_GICP/{dataset}/manuais_refinadas/pose_{i+1}_{i}.txt") for i in range(n_nuvens-1)]
    loopclosure = np.loadtxt(f"relative_poses_FGR_GICP/{dataset}/manuais_refinadas/pose_0_{n_nuvens-1}.txt")
    relative_poses_FGR_GICP.append(loopclosure)
    # CALCULATE PAIRWISE RMSE and FITNESS
    RMSE_FGR, fitness_FGR = myf.calculate_RMSE_and_fitness(nuvens, relative_poses_FGR, voxel_size)
    RMSE_FGR_GICP, fitness_FGR_GICP = myf.calculate_RMSE_and_fitness(nuvens, relative_poses_FGR_GICP, voxel_size)
    # PLOT RMSE and FITNESS
    myf.plot_RMSE_BAR(RMSE_FGR,RMSE_FGR_GICP)
    myf.plot_Fitness_BAR(fitness_FGR,fitness_FGR_GICP)
    # Compose relative poses to obtain absolute poses
    absolute_poses_FGR = myf.poses_relativas_para_absolutas(relative_poses_FGR)
    absolute_poses_FGR_GICP = myf.poses_relativas_para_absolutas(relative_poses_FGR_GICP)
    # Apply absolute poses to draw before and after (all pairs must be correctly registered) 
    print("Showing 3D reconstruction with FGR only")
    myf.apply_poses_in_clouds(absolute_poses_FGR, nuvens)
    print("Showing 3D reconstruction with FGR+M-GICP refinement")
    myf.apply_poses_in_clouds(absolute_poses_FGR_GICP, nuvens)
    # Print the enhancement in percetage of the RMSE and Fitness for all poses
    print(f"Percentage of reduction in RMSE: {1-(np.mean(RMSE_FGR_GICP)/np.mean(RMSE_FGR))}")
    print(f"Percentage of increase in sobreposition: {1-(np.mean(fitness_FGR)/np.mean(fitness_FGR_GICP))}")


    
# REGISTRATION LOOP - IF YOU CHOSE TO REPRODUCE RESULTS
elif decision == '0': 
    # PART 1 - FGR PAIRWISE REGISTRATION IN ALL PAIRS UNTIL THE CIRCUIT IS CLOSED
    results_FGR = []
    times_FGR = []
    # FGR registration loop:
    for i in range(n_nuvens):
        if i < n_nuvens-1:
            # Odometry case: 2 => 1, 3 => 2, n+1 => n
            print(f"Registering cloud {i+1} in cloud {i} with FGR")
            ini = time.time()
            result_FGR = myf.registro_FGR(nuvens[i+1], nuvens[i], voxel_size)
            time_FGR = time.time()-ini
            print(f"Pair {i+1}->{i} time taken: {round(time_FGR)} sec")
            print(f"Pair {i+1}->{i} fitness: {result_FGR.fitness}")
            print(f"Pair {i+1}->{i} RMSE: {result_FGR.inlier_rmse}")
            times_FGR.append(time_FGR)
            results_FGR.append(result_FGR)
            # Draw registration result
            myf.desenhar_resultado_registro(nuvens[i+1], nuvens[i], result_FGR.transformation)
        elif i == n_nuvens-1:
            # Loop-closure case: first_cloud => last_cloud
            print(f"Registering cloud 0 in cloud {i} with FGR")
            ini = time.time()
            result_FGR = myf.registro_FGR(nuvens[0], nuvens[i], voxel_size)
            time_FGR = time.time()-ini
            print(f"Pair 0->{i} time taken: {round(time_FGR)} sec")
            print(f"Pair 0->{i} fitness: {result_FGR.fitness}")
            print(f"Pair 0->{i} RMSE: {result_FGR.inlier_rmse}")
            times_FGR.append(time_FGR)
            results_FGR.append(result_FGR)
            # Draw registration result
            myf.desenhar_resultado_registro(nuvens[0], nuvens[i], result_FGR.transformation)
    
    # Plot times taken
    print(f"Time taken by FGR in all pairs: {sum(np.round(times_FGR))} sec")
    myf.plot_bar_time(times_FGR)
    
    
    # Calculate absolute poses and apply them.
    relative_poses_FGR = [results_FGR[i].transformation for i in range(n_nuvens)]
    absolute_poses_FGR = myf.poses_relativas_para_absolutas(relative_poses_FGR[0:n_nuvens-1])
    myf.apply_poses_in_clouds(absolute_poses_FGR, nuvens)
    
    
    # PART 2 - APPLY M-GICP REFINEMENT IN ALL PAIRS
    times_GICP = []
    results_GICP = []
    initial_Ts = results_FGR
    # Multiscale-GICP loop
    for i in range(n_nuvens):
        # Odometry case: 2 => 1, 3 => 2, n+1 => n
        if i < n_nuvens-1:
            print(f"Registering cloud {i+1} in cloud {i} with M-GICP")
            ini = time.time()
            result_GICP, _ = myf.multiscale_GICP(nuvens[i+1], nuvens[i], voxel_size, initial_Ts[i].transformation)
            time_GICP = time.time()-ini
            print(f"Pair {i+1}->{i} time taken: {round(time_GICP)} sec")
            print(f"Pair {i+1}->{i} fitness: {result_GICP.fitness}")
            print(f"Pair {i+1}->{i} RMSE: {result_GICP.inlier_rmse}")
            times_GICP.append(time_GICP)
            results_GICP.append(result_GICP)
            # Draw registration result
            myf.desenhar_resultado_registro(nuvens[i+1], nuvens[i], result_GICP.transformation)
        # Loop-closure case: first_cloud => last_cloud
        elif i == n_nuvens-1:
            print(f"Registering cloud 0 in cloud {i} with M-GICP")
            ini = time.time()
            result_GICP, _ = myf.multiscale_GICP(nuvens[0], nuvens[i], voxel_size, initial_Ts[i].transformation)
            time_GICP = time.time()-ini
            print(f"Pair 0->{i} time taken: {round(time_GICP)} sec")
            print(f"Pair 0->{i} fitness: {result_GICP.fitness}")
            print(f"Pair 0->{i} RMSE: {result_GICP.inlier_rmse}")
            times_GICP.append(time_GICP)
            results_GICP.append(result_GICP)
            # Draw registration result
            myf.desenhar_resultado_registro(nuvens[0], nuvens[i], result_GICP.transformation)
    
    print(f"Time taken by multi-scale GICP in all pairs: {round(sum(times_GICP))} sec")
    myf.plot_bar_time(times_GICP)
    
    
    # Calculate absolute poses with relative poses and apply them.
    relative_poses_GICP = [results_GICP[i].transformation for i in range(n_nuvens)]
    absolute_poses_GICP = myf.poses_relativas_para_absolutas(relative_poses_GICP)
    myf.apply_poses_in_clouds(absolute_poses_GICP, nuvens)


'''
# SAVE PAIRWISE TRANSFORMATIONS (FGR coarse and GICP refined relative poses)
for i in range(n_nuvens):
    if i < n_nuvens-1:
        # Odometric relative poses:
        np.savetxt(f"relative_poses_FGR/{dataset}/pose_{i+1}_{i}.txt", results_FGR[i].transformation, fmt="%.10f")
        np.savetxt(f"relative_poses_FGR_GICP/{dataset}/pose_{i+1}_{i}.txt", results_GICP[i].transformation, fmt="%.10f")
    elif i == n_nuvens-1:
        # Loop-closure relative pose:
        np.savetxt(f"relative_poses_FGR/{dataset}/pose_0_{i}.txt", results_FGR[i].transformation, fmt="%.10f")
        np.savetxt(f"relative_poses_FGR_GICP/{dataset}/pose_0_{i}.txt", results_GICP[i].transformation, fmt="%.10f")
'''

# TODO: Add Trees dataset