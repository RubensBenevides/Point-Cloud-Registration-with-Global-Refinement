import open3d as o3d
import quaternion as quat
import numpy as np
import copy
import time
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')
import seaborn as sns
import pandas as pd



# SCRIPT FOR APPLYING GLOBAL REFINEMENT MODELS (GRM) IN THE NCLT DATASET
# GRMs: LUM, SLERP, SLERP+LUM (ours), and g2o (GRISETTI, Giorgio et al.)



def plot_bar_time(time_list):
    rotulos = ['LUM','SLERP','SLERP+LUM','g2o']
    d = {'GRM': rotulos, 'Time (seg)': time_list}
    sns.barplot(data = pd.DataFrame(d), x='GRM', y='Time (seg)')
    plt.show()


# Function to invert a pose. INPUT: 4x4 matrix. OUTPUT: 4x4 matrix.
def Invert_pose(T_4x4):
    R_inv = np.transpose(T_4x4[0:3,0:3])
    t_inv = np.transpose([-R_inv@T_4x4[0:3,3]])
    T_inv = np.vstack((np.hstack((R_inv,t_inv)), np.array([0,0,0,1])))
    return T_inv


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


# Function that, given n absolute poses, applies them to each cloud n in the list of n clouds
# The first pose on the list should be the identity. 
# The first cloud in the list will be the global origin. 
# INPUT: lista_poses = list of absolute poses; lista_nuvens = list of clouds.
# OUTPUT: Open the open3d window and draw results. 
def apply_poses_in_clouds(lista_poses, lista_nuvens, lines):
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
    vis.add_geometry(lines)
    vis.run()
    vis.destroy_window()


# Funcao que reorna a trajetoria das poses.
# ENTRADA: poses absolutas (python list).
# SAIDA: Conjunto de linhas (objeto da Open3D).
def criar_trejetoria_com_linhas(poses):
    # Ler translacoes e rotacoes das poses e stackear tudo verticalmente
    matriz_translacoes = np.zeros((1,3))
    matriz_rotacoes = np.identity(3)
    for i in range(len(poses)-1):
        t = np.vstack((matriz_translacoes, np.transpose(np.transpose([poses[i][:3,3]]))))
        R = np.vstack((matriz_rotacoes, poses[i][:3,:3]))
        matriz_translacoes = t
        matriz_rotacoes = R
    # Inicializar lista das linhas e dos eixos:
    linhas = []
    for i in range(len(matriz_translacoes)):
        # Definir pares de pontos que definem as linhas:
        if i < len(matriz_translacoes)-1:
            linha = [i,i+1]
            linhas.append(linha)
        elif i == len(matriz_translacoes)-1:
            linha = [i,0]
            linhas.append(linha)
    # Criar conjunto de linhas, essa funcao retorna tudo numa so geometria
    conjunto_linhas = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(matriz_translacoes),
                                            lines=o3d.utility.Vector2iVector(linhas))
    return conjunto_linhas


# Function which calculates RMSE and Fitness given a list of poses, a list of clouds and a distance.
# IMPUT: 'lista_nuvens' = list of n clouds. 'T_circuito' = n-1 relative poses of the circuit.
# 'distancia' = maximum matching distance to consider when calculating the RMSE and fitness.
# If the nº of poses is equal to nº of clouds the circuit is considered closed, that is, 
# the last pose is considered to be the loop-closure pose (last->first).
# OUTPUT: 2 python lists, one of RMSE and other of Fitness, to each pair in the circuit.
def calculate_RMSE_and_fitness(lista_nuvens, T_circuito, distancia):
    n_nuvens = len(lista_nuvens)
    n_T = len(T_circuito)
    lista_RMSE, lista_fitness = [],[]
    # Caso a pose loopclosure seja passada: n_nuvens == n_T (poses_relativas)
    if n_nuvens == n_T:
        for i in range(n_nuvens):
            if i < n_nuvens-1:
                result = o3d.pipelines.registration.evaluate_registration(lista_nuvens[i+1], lista_nuvens[i], distancia, T_circuito[i])
                lista_RMSE.append(result.inlier_rmse)
                lista_fitness.append(result.fitness)
            elif i == n_nuvens-1:
                # Avaliacao da pose loopclosure (0 -> n)
                result = o3d.pipelines.registration.evaluate_registration(lista_nuvens[0], lista_nuvens[i], distancia, T_circuito[i])
                lista_RMSE.append(result.inlier_rmse)
                lista_fitness.append(result.fitness)
    # Caso a quantidade n_nuvens seja uma unidade maior que a quantidade de poses, entao o circuito nao eh fechado
    elif n_nuvens-1 == n_T:
        for i in range(n_T):
            result = o3d.pipelines.registration.evaluate_registration(lista_nuvens[i+1], lista_nuvens[i], distancia, T_circuito[i])
            lista_RMSE.append(result.inlier_rmse)
            lista_fitness.append(result.fitness)
    else: print("The number of clouds and poses are inconsistent")
    return lista_RMSE, lista_fitness


# Function to organize weights in diagonal matrix  
def Montar_Matriz_Diagonal_Pesos(Pesos,n_nuvens):
    # Pesos deve ser do tipo lista 
    P = Pesos[0]*np.ones(3)
    for i in range(1,n_nuvens):
        P_aux = np.ones(3)*Pesos[i]
        P = np.hstack((P,P_aux))
    return np.diagflat(P)


# Function to organize vector of observations using relative poses
# INPUT: T_circuito = list of relative poses; 
# lista_rotacoes_origem = list of absolute rotations
# OUTPUT: list of obvervations in a numpy array + closure error in translations 
def Montar_Vetor_Lb_translacoes(T_circuito,lista_rotacoes_origem):
    # Montar vetor Lb de 3*(n_nuvens) linhas por 1 coluna: 
    Lb = T_circuito[0][:3,3]
    LoopClosure_Translacao = T_circuito[0][:3,3]
    for i in range(len(T_circuito)-1):
        aux_Lb = lista_rotacoes_origem[i]@T_circuito[i+1][:3,3]
        LoopClosure_Translacao = LoopClosure_Translacao + aux_Lb
        Lb = np.hstack([Lb,aux_Lb]) # vetor de Lb com as translacoes rotacionadas
    Lb = np.reshape(Lb,(len(Lb),1)) # organiza o vetor em uma coluna
    return [Lb,LoopClosure_Translacao]


# Function that composes rotations to the origin in forward and reverse order and interpolates them
# by SLERP. INPUT: list of quaternions that represent rotations between pairs of clouds in a closed
# circuit (relative rotations). OUTPUT: list of interpolated quaternions that represent absolute
# rotations with respect to the global origin. The last quaternion in the list is the closure error,
# ideally, it would equal the identity of the quaternions (1,0,0,0), but only if the circuit is 
# error-free, as it accumulates all rotations back to origin. 
def Ajustamento_Quaternios_SLERP(lista_quat):
    # Initialize list that will receive the quaternions that represent absolute (global) rotations
    lista_quat_origem = []
    # Loop that accumulates rotations to the origin
    for i in range(len(lista_quat)):
        # Initialize rotations with null rotation (identity)
        ri = quat.from_rotation_matrix(np.identity(3))
        # Loop that accumulates j rotations
        for j in range (len(lista_quat)-i-1,-1,-1): # iterates j from top to bottom (n, n-1, ...)
            ri = ri*lista_quat[j] # ri accumulates rotations multiplicatively (..., r2 = q2*q1, r1 = q1)
        lista_quat_origem.append(ri) # Add the absolute rotation to lista_quat_origem
    # Inverts the list order, the last rotation becomes the one that accumulates all
    lista_quat_origem = list(reversed(lista_quat_origem))
    # Get previous list in reverse order of multiplication. It does not multiply everything again, 
    # just multiply by the inverse of which accumulates all (a trick from quaternions).
    quat_LoopClosure = lista_quat_origem[len(lista_quat)-1] # last of the list
    lista_quat_origem_inv = [lista_quat_origem[i]*quat_LoopClosure**(-1) for i in range(len(lista_quat)-1)]
    # Apply SLERP between the two lists of quaternions
    lista_quat_ajustado = []
    for i in range(len(lista_quat)-1):
        qi_ajustado = quat.quaternion_time_series.slerp(lista_quat_origem[i],lista_quat_origem_inv[i],0,1,t_out=(i+1)/len(lista_quat))
        lista_quat_ajustado.append(qi_ajustado)
    return [lista_quat_ajustado, quat_LoopClosure]


# Function to compose relative poses and ADJUST TRANSLATIONS ONLY (Lu & Milios, 1992).
# INPUT: T_circuito = list of relative poses: T10, T21, T32, ... Tn_n-1
# OUTPUT: List of absolute poses: T10, T20, T30, ... Tn_0.
# The last pose is the closure error (composition of all relative poses).
def reconstruir_Ts_para_origem_LUM(T_circuito,Pesos):
    # obter lista de rotacoes para a origem por composicao multiplicativa
    lista_rotacoes_origem = []
    for i in range(len(T_circuito)):
        LoopClosure = np.identity(3)
        for j in range (len(T_circuito)-i-1,-1,-1):        
            LoopClosure = LoopClosure@T_circuito[j][:3,:3] # acumula j matrizes rotacoes
        lista_rotacoes_origem.append(LoopClosure) # primeira rotacao da lista eh a LoopClosure
    lista_rotacoes_origem = list(reversed(lista_rotacoes_origem)) # inverte a lista
    # AJUSTAMENTO DAS TRANSLACOES ROTACIONADAS SEGUNDO (LU & MILIOS, 1998)
    # Montar vetor Lb de 3*(len(T_circuito)) linhas em 1 coluna:
    Lb,Translacao_LoopClosure = Montar_Vetor_Lb_translacoes(T_circuito,lista_rotacoes_origem)
    # Matriz digonal dos Pesos P:
    P = Montar_Matriz_Diagonal_Pesos(Pesos,len(T_circuito))
    # Montar matriz A com 3*(len(T_circuito)) linhas e 3*(len(T_circuito)-1) colunas
    A = np.diagflat([-np.ones((len(T_circuito)-1)*3)],-3)  # cria uma matriz com -1 na diagonal -3
    A = np.delete(A,[np.arange(len(A)-3,len(A))],1) # remove as 3 ultimas colunas de A
    np.fill_diagonal(A, 1.0)                        # preenche a diagonal principal com 1  
    AtPA = np.transpose(A)@P@A
    N = np.linalg.inv(AtPA)              # Matriz N = inv(A'PA)
    U = np.transpose(A)@P@Lb             # Matriz U = A'Lb
    X = N@U                              # Matriz X de translacoes ajustadas para a origem
    V = -A@X+Lb                          # Residuo. O mesmo que sum(Lb) a soma das translacoes
    print(f"Sigma_Posterior = Vt*P*V of LUM = {(np.transpose(V)@P@V)/3} ")
    # Juntar (rotacao + translacao ajustada LUM) em T(4x4): 
    Poses_ajustadas_LUM = []
    aux_0001 = np.array([0,0,0,1])
    for i in range(len(T_circuito)-1):
        Pose = np.vstack((np.hstack((lista_rotacoes_origem[i], X[3*i:3*(i+1)])), aux_0001))
        Poses_ajustadas_LUM.append(Pose)
    # Insert identity pose:
    Poses_ajustadas_LUM.insert(0,np.identity(4))
    return Poses_ajustadas_LUM


# Function to compose relative poses and ADJUST ROTATIONS ONLY
# INPUT: T_circuito = list of relative poses: T10, T21, T32, ... Tn_n-1
# OUTPUT: List of absolute poses: T10, T20, T30, ... Tn_0.
# The last pose is the closure error (composition of all relative poses).
def reconstruir_Ts_para_origem_SLERP(T_circuito):
    # obter rotacoes e translacoes do circuito:
    lista_translacoes = [T_circuito[i][0:3,3] for i in range(len(T_circuito))]
    lista_quat = [quat.from_rotation_matrix(T_circuito[i][0:3,0:3]) for i in range(len(T_circuito))]
    # AJUSTAMENTO DOS QUATERNIOS POR SLERP
    lista_quat_ajustado, quat_LoopClosure = Ajustamento_Quaternios_SLERP(lista_quat)
    # Obter as translacoes de cada nuvem para a origem
    t_ini = lista_translacoes[0]
    lista_translacoes_origem = [lista_translacoes[0]]
    for i in range(len(T_circuito)-1):
        t_posterior = quat.as_rotation_matrix(lista_quat_ajustado[i])@lista_translacoes[i+1] + t_ini
        t_ini = t_posterior # t10 recebe t20 e continua
        lista_translacoes_origem.append(t_posterior)
    # Juntar (rotacoes slerpadas + translacoes) em pose T(4x4):
    Poses_SLERP = []
    aux_0001 = np.array([0,0,0,1])
    for i in range(len(T_circuito)-1):
        Matriz_R = quat.as_rotation_matrix(lista_quat_ajustado[i]) # Quaternios -> Matriz
        Pose = np.vstack((np.hstack((Matriz_R, np.transpose([lista_translacoes_origem[i]]))), aux_0001))
        Poses_SLERP.append(Pose)
    # Insert identity pose:
    Poses_SLERP.insert(0,np.identity(4))
    return Poses_SLERP


# Function to compose relative poses and ADJUST TRANSLATIONS AND ROTATIONS.
# INPUT: T_circuito = list of relative poses: T10, T21, T32, ... Tn_n-1.
# Pesos = list of weigths to use in (Lu & Milios, 1992) adjustment.
# OUTPUT: List of absolute poses: T10, T20, T30, ... Tn_0.
# The last pose is the closure error (composition of all relative poses).
def reconstruir_Ts_para_origem_SLERP_LUM(T_circuito,Pesos):
    # Obter rotacoes e translacoes do circuito
    lista_quat = [quat.from_rotation_matrix(T_circuito[i][0:3,0:3]) for i in range(len(T_circuito))]
    # AJUSTAMENTO DOS QUATERNIOS POR SLERP
    lista_quat_ajustado, quat_LoopClosure = Ajustamento_Quaternios_SLERP(lista_quat)
    # AJUSTAMENTO DAS TRANSLACOES ROTACIONADAS SEGUNDO (LU & MILIOS, 1998)
    # Retornar de quaternios para matrizes de rotacao quat -> Matrix
    lista_R_origem = [quat.as_rotation_matrix(lista_quat_ajustado[i]) for i in range(len(T_circuito)-1)]
    # Montar vetor Lb: 3*(len(T_circuito)-1)+3 observacoes em uma coluna:
    Lb, Translacao_LoopClosure = Montar_Vetor_Lb_translacoes(T_circuito,lista_R_origem)
    # Matriz diagonal dos Pesos P [(3*len(T_circuito))x(3*len(T_circuito))]:
    P = Montar_Matriz_Diagonal_Pesos(Pesos,len(T_circuito))
    # Montar matriz A [(3*len(T_circuito)) x 3*(len(T_circuito)-1)]
    A = np.diagflat([-np.ones((len(T_circuito)-1)*3)],-3)  # cria uma matriz com -1 na diagonal -3
    A = np.delete(A,[np.arange(len(A)-3,len(A))],1) # remove as 3 ultimas colunas de A
    np.fill_diagonal(A, 1.0)                        # preenche a diagonal principal com 1's  
    AtPA = np.transpose(A)@P@A
    N = np.linalg.inv(AtPA)              # Matriz N = inv(A'PA) 
    U = np.transpose(A)@P@Lb             # Matriz U = A'Lb
    X = N@U                              # Translacoes ajustadas para a origem
    V = -A@X+Lb                          # Residuo. O mesmo que sum(Lb) a soma das translacoes
    print(f"Sigma_Posterior = Vt*P*V of LUM = {np.transpose(V)@P@V/3} GL = 3")
    # Juntar (rotacao slerpada + translacao ajustada LUM) em uma pose T(4x4): 
    Lista_Poses_Ajustadas_SLERP_LUM = []
    aux_0001 = np.array([0,0,0,1])
    for i in range(len(T_circuito)-1):
        Pose = np.vstack(( np.hstack((lista_R_origem[i], X[i*3:3*(i+1)])), aux_0001)) 
        Lista_Poses_Ajustadas_SLERP_LUM.append(Pose)
    # Insert identity pose:
    Lista_Poses_Ajustadas_SLERP_LUM.insert(0,np.identity(4))
    return Lista_Poses_Ajustadas_SLERP_LUM


# 0 - LOAD NCLT POINT CLOUDS
n_clouds = 901
clouds = [o3d.io.read_point_cloud(f"nuvens/nuvens_pre_processadas/NCLT/s{i}.pcd") for i in range(n_clouds)]


# 1 - INITIALIZE POSE_GRAPH AND ADD IDENTITY TO THE FIRST VERTEX
pose_graph = o3d.pipelines.registration.PoseGraph()
pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))


# 2 - LOAD ALREADY REFINED ABSOLUTE POSES (FGR+M-GICP)
absolute_poses_FGR_GICP = [np.loadtxt(f"absolute_poses_FGR_GICP/NCLT/pose{i}.txt") for i in range(n_clouds)]


# 3 - LOAD RELATIVE POSES TO BUILD THE POSE_GRAPH (NECESSARY TO DO G2O OPTIMIZATION) 
relative_poses_FGR_GICP = [np.loadtxt(f"relative_poses_FGR_GICP/NCLT/pose_{i+1}_{i}.txt") for i in range(n_clouds-1)]
loop_closure = np.loadtxt(f"relative_poses_FGR_GICP/NCLT/pose_0_900.txt")
relative_poses_FGR_GICP.append(loop_closure)


# 4 - BUILD POSE_GRAPH
voxel_size = 0.1
for i in range(n_clouds):
    # Poses sequenciais (odometric)
    if i < n_clouds-1:
        # Add absolute pose as an vertex of the pose_graph 
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(absolute_poses_FGR_GICP[i]))
        # Get inverted relative pose (n -> n+1)
        relative_pose = Invert_pose(relative_poses_FGR_GICP[i])
        # Get information matrix
        information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(clouds[i],
            clouds[i+1], 
            voxel_size, 
            relative_pose)
        # Add relative pose and information matrix in the edge of the pose_graph (edge i->i+1)
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i,
            i+1,
            relative_pose,
            information_matrix,
            uncertain=False)) # False means that this edge is not a loop-closure edge
    elif i == n_clouds-1:
        # Add the last relative pose which closes the circuit in a loop
        relative_pose = Invert_pose(relative_poses_FGR_GICP[i])
        # Get information matrix
        information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(clouds[i],
            clouds[0], 
            voxel_size, 
            relative_pose)
        # Add relative pose and information matrix in the edge of the pose_graph (edge {i<->i+1})
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i,
            0,
            relative_pose,
            information_matrix,
            uncertain=True)) # True means that this edge is a loop-closure edge


# 5 - DEFINE PARAMETERS FOR GLOBAL G2O OPTIMIZATION 
voxel_size = 0.1
option = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance = 2*voxel_size, 
    edge_prune_threshold = 0.25, # Default value
    reference_node = 0)          # The same used in SLERP+LUM


# 6 - OPTIMIZE THE POSE_GRAPH WITH G2O
times_g2o = []
for i in range(10):
    start = time.time()
    o3d.pipelines.registration.global_optimization(pose_graph, 
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(), 
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), 
        option)
    end = time.time()
    time_g2o = end - start
    times_g2o.append(time_g2o)

print(f"Mean time of 10 g2o optimizations: {np.mean(times_g2o)}")


# 7 - RECOVER ABSOLUTE POSES FROM THE GRAPH
absolute_poses_g2o = [pose_graph.nodes[i].pose for i in range(n_clouds)]


# 8 - APPLY SLERP+LUM GLOBAL OPTIMIZATIONS ON RELATIVE POSES
# We use normalized RMSE [0,1] as weigths to our model.
RMSE, fitness = calculate_RMSE_and_fitness(clouds, relative_poses_FGR_GICP, 0.1)
weights_RMSE = ( np.asarray(RMSE) - min(RMSE) )/( max(RMSE) - min(RMSE) )
weights_ones = np.ones(n_clouds)
# Our model SLERP+LUM take as imput relative poses and output global refined absolute poses
times_LUM = []
times_SLERP = []
times_SLERP_LUM = []
for i in range(10):
    start = time.time()
    absolute_poses_LUM  = reconstruir_Ts_para_origem_LUM(relative_poses_FGR_GICP, weights_ones)
    end = time.time()
    time_LUM = end - start
    start = time.time()
    absolute_poses_SLERP = reconstruir_Ts_para_origem_SLERP(relative_poses_FGR_GICP)
    end = time.time()
    time_SLERP = end - start
    start = time.time()
    absolute_poses_SLERP_LUM = reconstruir_Ts_para_origem_SLERP_LUM(relative_poses_FGR_GICP, weights_RMSE)
    end = time.time()
    time_SLERP_LUM = end - start
    times_LUM.append(time_LUM)
    times_SLERP.append(time_SLERP)
    times_SLERP_LUM.append(time_SLERP_LUM)

print(f"Mean time of 10 LUM optimization: {np.mean(times_LUM)}")
print(f"Mean time of 10 SLERP optimization: {np.mean(times_SLERP)}")
print(f"Mean time of 10 SLERP_LUM optimization: {np.mean(times_SLERP_LUM)}")

# Plot bar time
time_list = [np.mean(times_LUM), np.mean(times_SLERP), np.mean(times_SLERP_LUM), np.mean(times_g2o)]
plot_bar_time(time_list)


# 9 - LOAD GROUNDTRUTH AND CALCULATE DIFFERENCES BETWEEN ALL ESTIMATED POSES
poses_groundtruth = [np.loadtxt(f"groundtruth/NCLT/pose{i}.txt") for i in range(n_clouds)]
dist_R_1, dist_t_1 = subtract_squared_poses(poses_groundtruth, absolute_poses_LUM)
dist_R_2, dist_t_2 = subtract_squared_poses(poses_groundtruth, absolute_poses_SLERP)
dist_R_3, dist_t_3 = subtract_squared_poses(poses_groundtruth, absolute_poses_SLERP_LUM)
dist_R_4, dist_t_4 = subtract_squared_poses(poses_groundtruth, absolute_poses_g2o)


# 10 - PLOT DIFERENCES - TRANSLATION
plt.plot(dist_t_1, label = "LUM ", color = (0,1,0))
plt.plot(dist_t_2, label = "SLERP", color = (1,0,0))
plt.plot(dist_t_3, label = "SLERP+LUM (ours)", color = (0,0,1))
plt.plot(dist_t_4, label = "g2o", color = (1,1,0))
# name the x,y axis
plt.xlabel('Absolute poses')
plt.ylabel('Error (m)')
# Show the legend and grid
plt.grid(True)
plt.legend()
plt.show()


# 11 - PLOT DIFERENCES - ROTATION
plt.plot(dist_R_1, label = "LUM")
plt.plot(dist_R_2, label = "SLERP")
plt.plot(dist_R_3, label = "SLERP+LUM (ours)")
plt.plot(dist_R_4, label = "g2o")
# name the x,y axis
plt.xlabel('Absolute poses')
plt.ylabel('Error [0,1] (adim.)')
# Show the legend and grid
plt.grid(True)
plt.legend()
plt.show()


# 12 - DRAW RESULTS
linhas_groundtruth = criar_trejetoria_com_linhas(poses_groundtruth)
linhas_LUM = criar_trejetoria_com_linhas(absolute_poses_LUM)
linhas_SLERP = criar_trejetoria_com_linhas(absolute_poses_SLERP)
linhas_SLERP_LUM = criar_trejetoria_com_linhas(absolute_poses_SLERP_LUM)
linhas_g2o = criar_trejetoria_com_linhas(absolute_poses_g2o)

apply_poses_in_clouds(absolute_poses_LUM, clouds, linhas_LUM)
apply_poses_in_clouds(absolute_poses_SLERP, clouds, linhas_SLERP)
apply_poses_in_clouds(absolute_poses_SLERP_LUM, clouds, linhas_SLERP_LUM)
apply_poses_in_clouds(absolute_poses_g2o, clouds, linhas_g2o)


# 13 - PAINT LINES
# LUM = GREEN | SLERP = RED | SLERP+LUM = BLUE | G2O = YELLOW | GROUNDTRUTH = WHITE
linhas_LUM.paint_uniform_color([1,0,0])
linhas_SLERP.paint_uniform_color([0,1,0])
linhas_SLERP_LUM.paint_uniform_color([0,0,1])
linhas_g2o.paint_uniform_color([1,1,0])
linhas_groundtruth.paint_uniform_color([0,0,0])


def draw_circuit_lines(list_of_line_sets):
    n = len(list_of_line_sets)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width = 1920, height = 1080)
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=60.0)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.line_width = 5.0
    for i in range(n):
        vis.add_geometry(list_of_line_sets[i])
    vis.run()
    vis.destroy_window()
        

list_line_sets = [linhas_LUM,linhas_SLERP,linhas_SLERP_LUM,linhas_g2o,linhas_groundtruth]
draw_circuit_lines(list_line_sets)


'''
# SAVE ALL TRANSFORMATIONS
for i in range(n_clouds):
    np.savetxt(f"absolute_poses_LUM/NCLT/pose{i}.txt", absolute_poses_LUM[i], fmt="%.10f")
    np.savetxt(f"absolute_poses_SLERP/NCLT/pose{i}.txt", absolute_poses_SLERP[i], fmt="%.10f")
    np.savetxt(f"absolute_poses_SLERP_LUM/NCLT/pose{i}.txt", absolute_poses_SLERP_LUM[i], fmt="%.10f")
'''
