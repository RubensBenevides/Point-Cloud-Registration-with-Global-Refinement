import copy
import numpy as np
import open3d as o3d
import time 
from matplotlib import pyplot as plt


# Function to show the result of registration with tensors
def draw_registration_result(source, target, transformation):
    source_temp = source.clone()
    target_temp = target.clone()
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp.to_legacy(), target_temp.to_legacy()])


# Function to show the result of registration with defout open3d object
def draw_registration_result_2(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def plot_rmse(registration_result):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11, 3))
    axes.set_title("RMSE x Iteration")
    axes.plot(registration_result.loss_log["index"].numpy(),
              registration_result.loss_log["inlier_rmse"].numpy())
    plt.show()


def plot_rmse_2(lista):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11, 3))
    axes.set_title("RMSE vs Iteração")
    axes.plot([i for i in range(len(lista))], lista)
    plt.show()


def plot_scale_wise_rmse(registration_result):
    scales = registration_result.loss_log["scale"].numpy()
    iterations = registration_result.loss_log["iteration"].numpy()
    num_scales = scales[-1][0] + 1
    fig, axes = plt.subplots(nrows=1, ncols=num_scales, figsize=(13, 5))
    fig.tight_layout()
    masks = {}
    for scale in range(0, num_scales):
        masks[scale] = registration_result.loss_log["scale"] == scale
        rmse = registration_result.loss_log["inlier_rmse"][masks[scale]].numpy()
        iteration = registration_result.loss_log["iteration"][masks[scale]].numpy()
        title_prefix = "Escala" + str(scale)
        axes[scale].set_title(title_prefix)
        axes[scale].plot(iteration, rmse)
    plt.show()


def plot_scale_wise_rmse_2(rmse_multiscale_GICP):
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(11, 3))
    fig.tight_layout()
    iteracoes_1 = [i for i in range(50)]
    iteracoes_2 = [i for i in range(40)]
    iteracoes_3 = [i for i in range(30)]
    iteracoes_4 = [i for i in range(20)]
    iteracoes_5 = [i for i in range(10)]
    rmse_1 = rmse_multiscale_GICP[0:50]
    rmse_2 = rmse_multiscale_GICP[50:90]
    rmse_3 = rmse_multiscale_GICP[90:120]
    rmse_4 = rmse_multiscale_GICP[120:140]
    rmse_5 = rmse_multiscale_GICP[140:150]
    axes[0].set_title("Escala 1")
    axes[0].plot(iteracoes_1, rmse_1)
    axes[1].set_title("Escala 2")
    axes[1].plot(iteracoes_2, rmse_2)
    axes[2].set_title("Escala 3")
    axes[2].plot(iteracoes_3, rmse_3)
    axes[3].set_title("Escala 4")
    axes[3].plot(iteracoes_4, rmse_4)
    axes[4].set_title("Escala 5")
    axes[4].plot(iteracoes_5, rmse_5)
    plt.show()


def plor_GICP_vs_multi_scale_GICP(RMSE_GICP, rmse_multiscale_GICP):
    # Plot both RMSE graphics
    plt.plot(rmse_multiscale_GICP, label = "GICP em Multi-escala")
    plt.plot(RMSE_GICP, label = "GICP normal")
    # name the x axis
    plt.xlabel('Iterações')
    # name the y axis
    plt.ylabel('RMSE')
    # show the legend
    plt.legend()
    plt.show()


def plot_multiscale_vs_normal_ICP(registration_icp, registration_ms_icp):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(13, 5))
    axes.set_title("Vanilla-ICP vs Multi-Scale-ICP, RMSE vs Iteration")
    if len(registration_ms_icp.loss_log["index"]) > len(registration_icp.loss_log["inlier_rmse"]):
        axes.plot(registration_ms_icp.loss_log["index"].numpy(),
                registration_ms_icp.loss_log["inlier_rmse"].numpy(),
                registration_icp.loss_log["inlier_rmse"].numpy())
    else:
        axes.plot(registration_icp.loss_log["index"].numpy(),
                registration_icp.loss_log["inlier_rmse"].numpy(),
                registration_ms_icp.loss_log["inlier_rmse"].numpy())
    plt.show()


# 1 - Importar par de nuvens
source = o3d.t.io.read_point_cloud("nuvens/nuvens_pre_processadas/Facade/s0.pcd")
target = o3d.t.io.read_point_cloud("nuvens/nuvens_pre_processadas/Facade/s1.pcd")


# 2 - Estimar nomais
source.estimate_normals(20)
target.estimate_normals(20)


# 3 - Criterios de parada
# Criterios de parada do ICP-normal (relative_fitness, relative_rmse, max_iteration):
criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(10**-6, 10**-6, 250)
# Criterios de parada do Multi-Scale-ICP (uma lista de criterios, um para cada escala) 
criteria_list = [o3d.t.pipelines.registration.ICPConvergenceCriteria(10**-2, 10**-2, 50),
                 o3d.t.pipelines.registration.ICPConvergenceCriteria(10**-3, 10**-3, 40),
                 o3d.t.pipelines.registration.ICPConvergenceCriteria(10**-4, 10**-4, 30),
                 o3d.t.pipelines.registration.ICPConvergenceCriteria(10**-5, 10**-5, 20),
                 o3d.t.pipelines.registration.ICPConvergenceCriteria(10**-6, 10**-6, 10)]


# 4 - Escalas
# Vetor de escala para o ICP-normal
voxel_size = 0.1
# Vetor de escalas para Multi-Scale-ICP
voxel_sizes = o3d.utility.DoubleVector([8*voxel_size, 4*voxel_size, 2*voxel_size, voxel_size, voxel_size/2])


# 5 - Distancias maximas para busca de correspondencias
# Para o ICP-normal:
max_correspondence_distance = 2*voxel_size
# Para o Multi-Scale-ICP:
max_correspondence_distances = o3d.utility.DoubleVector([4*voxel_sizes[0], 3*voxel_sizes[1], 2.5*voxel_sizes[2], 2*voxel_sizes[3], 2*voxel_sizes[4]])


# 6 - Transformacao Inicial para ambos:
init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)


# 7 - Selecionar funcao objetiva do ICP (ponto-a-plano)
estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()


# Setar verbosidade: mostrar o maximo de informacoes para fazer um bom fine-tunning
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)


# Salvar informacoes: fitness, inlier_rmse, iteracoes, etc. para analisar o resultado
save_loss_log = True


# 8 - Aplicar o ICP normal
s = time.time()
registration_icp = o3d.t.pipelines.registration.icp(source,
                                                    target,
                                                    max_correspondence_distance,
                                                    init_source_to_target, 
                                                    estimation, 
                                                    criteria,
                                                    voxel_size,
                                                    save_loss_log)
icp_time = time.time() - s


# 8.1 - Mostrar resultados: tempo, sobreposicao e RMSE dos pares inliers de acordo com a dist max. de corres.)
print("Time taken by ICP: ", icp_time)
print("Inlier Fitness: ", registration_icp.fitness)
print("Inlier RMSE: ", registration_icp.inlier_rmse)


# 9 - Aplicar o Multi-Scale-ICP
s = time.time()
registration_ms_icp = o3d.t.pipelines.registration.multi_scale_icp(source, 
                                                                   target,
                                                                   voxel_sizes,
                                                                   criteria_list,
                                                                   max_correspondence_distances,
                                                                   init_source_to_target, estimation,
                                                                   save_loss_log)
ms_icp_time = time.time() - s


# 9.1 - Mostrar resultados
print("Time taken by Multi-Scale ICP: ", ms_icp_time)
print("Inlier Fitness: ", registration_ms_icp.fitness)
print("Inlier RMSE: ", registration_ms_icp.inlier_rmse)


# 10 - Desenhar resultado (analise mais importante) dos dois registros
draw_registration_result(source, target, registration_icp.transformation)
draw_registration_result(source, target, registration_ms_icp.transformation)


# 11 - Grafico do ICP-normal
plot_rmse(registration_icp)


# 12 - Grafico do Multi-Scale-ICP
plot_rmse(registration_ms_icp)


# 13 - Grafico do Multi-Scale-ICP por iteracao
plot_scale_wise_rmse(registration_ms_icp)


# 14 - Comparacao do Grafico do Multi-Scale-ICP com o ICP-normal
plot_multiscale_vs_normal_ICP(registration_icp, registration_ms_icp)



# 15 - Compare GICP vs GICP multi-scale
# GICP sem multi escala
voxel_inicial = 0.1
source = o3d.io.read_point_cloud("nuvens/nuvens_nao_registradas/s0.pcd")
target = o3d.io.read_point_cloud("nuvens/nuvens_nao_registradas/s1.pcd")
source = source.voxel_down_sample(voxel_inicial)
target = target.voxel_down_sample(voxel_inicial)
source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_inicial, max_nn=20))
target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_inicial, max_nn=20))
inicial = np.identity(4)
lista_RMSE = []
iterations = 110
for i in range(iterations):
    result_icp = o3d.pipelines.registration.registration_generalized_icp(source,
                                                                        target,
                                                                        2*voxel_inicial,
                                                                        inicial,
                                                                        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                                                                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=i, relative_fitness = 10**-6, relative_rmse = 10**-6))
    lista_RMSE.append(result_icp.inlier_rmse)


# Escala 1
voxel_size = 5*voxel_inicial
source = o3d.io.read_point_cloud("nuvens/nuvens_nao_registradas/s0.pcd")
target = o3d.io.read_point_cloud("nuvens/nuvens_nao_registradas/s1.pcd")
source = source.voxel_down_sample(voxel_size)
target = target.voxel_down_sample(voxel_size)
source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=20))
target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=20))
iterations = 50
lista_RMSE_esc_1 = []
for i in range(iterations):
    result_icp_1 = o3d.pipelines.registration.registration_generalized_icp(source,
                                                                        target,
                                                                        5*voxel_size,
                                                                        inicial,
                                                                        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                                                                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=i, relative_fitness = 10**-6, relative_rmse = 10**-6))
    lista_RMSE_esc_1.append(result_icp_1.inlier_rmse)


# Escala 2
voxel_size = 2.5*voxel_inicial
source = o3d.io.read_point_cloud("nuvens/nuvens_nao_registradas/s0.pcd")
target = o3d.io.read_point_cloud("nuvens/nuvens_nao_registradas/s1.pcd")
source = source.voxel_down_sample(voxel_size)
target = target.voxel_down_sample(voxel_size)
source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=20))
target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=20))
lista_RMSE_esc_2 = []
iterations = 40
for i in range(iterations):
    result_icp_2 = o3d.pipelines.registration.registration_generalized_icp(source,
                                                                        target,
                                                                        4*voxel_size,
                                                                        result_icp_1.transformation,
                                                                        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                                                                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=i, relative_fitness = 10**-6, relative_rmse = 10**-6))
    lista_RMSE_esc_2.append(result_icp_2.inlier_rmse)


# Escala 3
voxel_size = voxel_inicial
source = o3d.io.read_point_cloud("nuvens/nuvens_nao_registradas/s0.pcd")
target = o3d.io.read_point_cloud("nuvens/nuvens_nao_registradas/s1.pcd")
source = source.voxel_down_sample(voxel_size)
target = target.voxel_down_sample(voxel_size)
source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=20))
target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=20))
lista_RMSE_esc_3 = []
iterations = 30
for i in range(iterations):
    result_icp_3 = o3d.pipelines.registration.registration_generalized_icp(source,
                                                                        target,
                                                                        3*voxel_size,
                                                                        result_icp_2.transformation,
                                                                        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                                                                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=i, relative_fitness = 10**-6, relative_rmse = 10**-6))
    lista_RMSE_esc_3.append(result_icp_3.inlier_rmse)


# Escala 4
voxel_size = voxel_inicial/2
source = o3d.io.read_point_cloud("nuvens/nuvens_nao_registradas/s0.pcd")
target = o3d.io.read_point_cloud("nuvens/nuvens_nao_registradas/s1.pcd")
source = source.voxel_down_sample(voxel_size)
target = target.voxel_down_sample(voxel_size)
source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=20))
target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=20))
lista_RMSE_esc_4 = []
iterations = 20
for i in range(iterations):
    result_icp_4 = o3d.pipelines.registration.registration_generalized_icp(source,
                                                                        target,
                                                                        2.5*voxel_size,
                                                                        result_icp_3.transformation,
                                                                        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                                                                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=i, relative_fitness = 10**-6, relative_rmse = 10**-6))
    lista_RMSE_esc_4.append(result_icp_4.inlier_rmse)


# Escala 5
voxel_size = voxel_inicial/4
source = o3d.io.read_point_cloud("nuvens/nuvens_nao_registradas/s0.pcd")
target = o3d.io.read_point_cloud("nuvens/nuvens_nao_registradas/s1.pcd")
source = source.voxel_down_sample(voxel_size)
target = target.voxel_down_sample(voxel_size)
source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=20))
target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=20))
lista_RMSE_esc_5 = []
iterations = 10
for i in range(iterations):
    result_icp_5 = o3d.pipelines.registration.registration_generalized_icp(source,
                                                                        target,
                                                                        2*voxel_size,
                                                                        result_icp_4.transformation,
                                                                        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                                                                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=i, relative_fitness = 10**-6, relative_rmse = 10**-6))
    lista_RMSE_esc_5.append(result_icp_5.inlier_rmse)


# 16 - Plot the RMSE of Multi-scale-GICP
rmse_multiscale_GICP = lista_RMSE_esc_1 + lista_RMSE_esc_2 + lista_RMSE_esc_3 + lista_RMSE_esc_4 + lista_RMSE_esc_5
plot_scale_wise_rmse_2(rmse_multiscale_GICP)
plot_rmse_2(rmse_multiscale_GICP)


# 17 - Plot the RMSE of GICP vs Multi-scale-GICP
plor_GICP_vs_multi_scale_GICP(lista_RMSE,rmse_multiscale_GICP)


# 18 - Show the registration with GICP and Multi-scale-GICP
draw_registration_result_2(source, target, result_icp.transformation)
draw_registration_result_2(source, target, result_icp_3.transformation)
