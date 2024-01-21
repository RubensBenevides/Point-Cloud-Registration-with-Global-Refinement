import open3d as o3d
import quaternion as quat
import numpy as np
np.set_printoptions(suppress=True)
import copy
import time
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')
import seaborn as sns
import pandas as pd


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


# Function to compose two relative poses. 
# INPUT: (T21,T10) homogeneous transformations.
# T21 = transformation which transform the system 2 to the reference system 1.
# T10 = transformation which transform the system 1 to the reference system 0.
# OUTPUT: T20 (transformation which transform the system 1 to the reference system 0).  
def Acumulate_Two_Poses(T21,T10):
    R21, t21 = T21[:3,:3], T21[:3,3]
    R10, t10 = T10[:3,:3], T10[:3,3]
    R20, t20 = R21@R10, R10@t21 + t10
    T20 = np.vstack((np.hstack((R20, np.reshape(t20,(3,1)))), np.array([0,0,0,1])))
    return T20


# Function to draw circuit path using a line_set as imput.
# IMPUT: line set geometry (Open3D object)
def draw_circuit_lines(list_of_line_sets):
    n = len(list_of_line_sets)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width = 1920, height = 1080)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.line_width = 5.0
    for i in range(n):
        vis.add_geometry(list_of_line_sets[i])
    vis.run()
    vis.destroy_window()


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


# Function that, given n absolute poses, applies them to each cloud in the list of clouds.
# The first pose on the list should be the identity. 
# The first cloud in the list will be the global origin. 
# INPUT: lista_poses = list of absolute poses; lista_nuvens = list of clouds.
# lines = circuit line set. OUTPUT: draw clouds and patch in Open3d window.
def apply_poses_in_clouds(lista_poses, lista_nuvens, lines):
    n_poses = len(lista_poses)
    # Make copies to preserv original clouds
    copias = copy.deepcopy(lista_nuvens)
    # Draw clouds
    vis = o3d.visualization.Visualizer()
    vis.create_window(width = 1920, height = 1080)
    # Aplicar poses nas copias
    for i in range(n_poses):
        copias[i].transform(lista_poses[i])
        vis.add_geometry(copias[i])
    vis.add_geometry(lines)
    vis.run()
    vis.destroy_window()


# Funcao que retorna a trajetoria das poses.
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


# Function to organize vector of observations using relative poses
# INPUT: T_circuito = list of relative poses; 
# lista_rotacoes_origem = list of absolute rotations
# OUTPUT: list of obvervations in a numpy array + closure error in translations 
def Montar_Vetor_Lb_translacoes(T_circuito,lista_rotacoes_origem):
    n = len(T_circuito)
    # Initialize identity
    absolute_translations = np.array([0,0,0])
    for i in range(n):
        # Vector of rotated translations
        t = lista_rotacoes_origem[i] @ T_circuito[i][0:3,3]
        # Stack
        absolute_translations = np.hstack((absolute_translations, t))
    # Reshape observation vector (Lb) as one column
    Lb = np.reshape(absolute_translations, (len(absolute_translations),1) )
    # Remove first three rows (identity)
    Lb = np.delete(Lb, (0,1,2), axis=0)
    return Lb


# Function that composes rotations to the origin in forward and reverse order and interpolates them
# by SLERP. INPUT: list of quaternions that represent rotations between pairs of clouds in a closed
# circuit (relative rotations) Last rotation should be the one to close the circuit. 
# OUTPUT: list of interpolated quaternions that represent absolute rotations, with respect to the 
# global origin. Optional: return the list of angular velocities of the quaternions. 
def Ajustamento_Quaternios_SLERP(relative_quat):
    n = len(relative_quat)
    # List that will receive absolute (global) rotations
    list_absolute_quat = []
    list_absolute_quat_inv = []
    absolute_quat = quat.from_rotation_matrix(np.identity(3))
    absolute_quat_reverse = quat.from_rotation_matrix(np.identity(3))
    # Transform relative quaternions into absolute quaternions by accumulating the rotations to global origin 
    for i in range(1,n):
        # Obtain rotations by composing quaternions in the direct way of the circuit
        absolute_quat = relative_quat[i-1]*absolute_quat # r1 = q1, r2 = q2*r1, r3 = q3*r2, ...
        # Obtain rotations by composing quaternions in the reverse way of the circuit
        # Note that q' = q**(-1), but (q_a*q_b) is different of (q_b*q_a)**(-1) 
        absolute_quat_reverse = absolute_quat_reverse*relative_quat[-i]
        absolute_quat_inv = absolute_quat_reverse**(-1)
        # Save list [q_1, q_2, ..., q_n]. Note that q_n = closure error in the direct way
        list_absolute_quat.append(absolute_quat)
        # Save list [q_n-1, q_n-2, ..., q_1]. Note that q_1 = closure error in the reverse way
        list_absolute_quat_inv.append(absolute_quat_inv)
    # Quaternions accumulated in reverse direction were saved in reverse order. To do the SLERP 
    # between both lists, we need one to grow in the opposite direction to the other.
    list_slerped_quat = []
    for i in range(1,n):
        # Now, we interpolate between the first and the second-to-last quaternio (ignore the closure error). 
        # Then, between the second quaternio and the third-to-last, and so on.
        slerped_quat = quat.quaternion_time_series.slerp(list_absolute_quat[i-1],
                                                         list_absolute_quat_inv[-i],
                                                         0, 1, t_out= i/n ) # interpolation interval 
        list_slerped_quat.append(slerped_quat)
    # Add the identity quaternion
    list_slerped_quat.insert(0, quat.from_rotation_matrix(np.identity(3)))
    return list_slerped_quat


# Function to compose relative poses and ADJUST TRANSLATIONS ONLY (Lu & Milios, 1992).
# INPUT: T_circuito = list of relative poses: T10, T21, T32, ... Tn_n-1
# OUTPUT: List of refined absolute poses: T10, T20, T30, ... Tn_0.
def reconstruir_Ts_para_origem_LUM(T_circuito):
    n = len(T_circuito)
    # Initialize list of absolute rotations and first absolute rotation
    list_absolute_R = []
    absolute_R = np.identity(3)
    # Compose relative R_3x3 into absolute R_3x3 by accumulating the rotations to global origin
    for i in range(n):
        list_absolute_R.append(absolute_R)
        absolute_R = T_circuito[i][0:3,0:3] @ absolute_R
    # Refinement of translation by (LU & MILIOS, 1998)
    # Make Lb vector. Array of [3*n, 1] rows, columns
    Lb = Montar_Vetor_Lb_translacoes(T_circuito, list_absolute_R)
    # Jacobian matrix A with [3*n, 3*(n-1)]. Three degrees of freedom.
    A = np.diagflat([-np.ones((n-1)*3)], -3)  # Define a matrix with -1's in the diagonal -3
    A = np.delete(A, [np.arange(len(A)-3, len(A))], 1) # Remove last 3 columns of A
    np.fill_diagonal(A, 1.0)                           # Fill the main diagonal with 1's  
    AtA = np.transpose(A)@A
    N = np.linalg.inv(AtA)               # Matrix N = inv(A'A)
    U = np.transpose(A)@Lb               # Matrix U = A'Lb
    X = N@U                              # Matrix Y (translacoes ajustadas para a origem)
    # Make 4x4 pose matrix (rotation + refined translation) 
    refined_poses_LUM = []
    aux_0001 = np.array([0,0,0,1])
    for i in range(1,n):
        Pose = np.vstack(( np.hstack(( list_absolute_R[i], X[3*(i-1):3*(i)] )), aux_0001))
        refined_poses_LUM.append(Pose)
    # Insert identity as the first pose:
    refined_poses_LUM.insert(0, np.identity(4))
    return refined_poses_LUM


# Function to compose relative poses and ADJUST ROTATIONS ONLY
# INPUT: T_circuito = list of relative poses: T10, T21, T32, ... Tn_n-1
# OUTPUT: List of absolute poses: T10, T20, T30, ... Tn_0.
def reconstruir_Ts_para_origem_SLERP(T_circuito):
    n = len(T_circuito)
    list_translations = []
    list_quaternions = []
    # Get relative rotations and translations from the circuit:
    for i in range(n):
        translation = T_circuito[i][0:3,3] 
        quaternion = quat.from_rotation_matrix(T_circuito[i][0:3,0:3])
        list_translations.append(translation)
        list_quaternions.append(quaternion)
    # Interpolate quaternions with SLERP (the optimization happens here)
    list_quat_slerped = Ajustamento_Quaternios_SLERP(list_quaternions)
    # Initialize list and auxiliar variables
    absolute_poses_slerped = []
    aux_0001 = np.array([0,0,0,1])
    t = np.array([[0,0,0]]) # Identity of translations = null translation
    for i in range(n):
        # Form the first absolute pose using the first translation and rotation
        slerped_R = quat.as_rotation_matrix(list_quat_slerped[i]) # Quaternion -> Matrix
        T_4x4 = np.vstack(( np.hstack(( slerped_R, np.transpose(t) )), aux_0001))
        # Save absolute pose
        absolute_poses_slerped.append(T_4x4)
        # Compose relative translations to absolute translations
        # t20 = R10@t21 + t10 | t30 = R20@t32 + t20 | t40 = R30@t43 + t30
        t = quat.as_rotation_matrix(list_quat_slerped[i])@list_translations[i] + t
    return absolute_poses_slerped


# Function to refine relative poses with SLERP+LUM model.
# INPUT: T_circuito = list of relative poses: T10, T21, T32, ... Tn_n-1.
# Pesos = list of weigths to use in (Lu & Milios, 1992) adjustment.
# OUTPUT: List of absolute poses: T10, T20, T30, ... Tn_0.
# The last pose is the closure error (composition of all relative poses).
def reconstruir_Ts_para_origem_SLERP_LUM(T_circuito):
    n = len(T_circuito)
    # Obter rotacoes e translacoes do circuito
    lista_quat = [quat.from_rotation_matrix(T_circuito[i][0:3,0:3]) for i in range(n)]
    # AJUSTAMENTO DOS QUATERNIOS POR SLERP
    lista_quat_ajustado = Ajustamento_Quaternios_SLERP(lista_quat)
    # Retornar de quaternios para matrizes de rotacao quat -> Matrix
    lista_R_origem = [quat.as_rotation_matrix(lista_quat_ajustado[i]) for i in range(n)]
    # Create Lb vector of 3*n observations in one column:
    Lb = Montar_Vetor_Lb_translacoes(T_circuito, lista_R_origem)
    # Create Jacobian matrix A (3*n)x(3*(n-1))
    A = np.diagflat([-np.ones((n-1)*3)],-3) # Make a matrix with -1's in the diagonal -3
    A = np.delete(A,[np.arange(len(A)-3,len(A))],1) # remove last 3 columns of A
    np.fill_diagonal(A, 1.0) # fill principal diagonal with 1's  
    AtA = np.transpose(A)@A
    N = np.linalg.inv(AtA)   # N = inv(A'A). Matrix of normal equations
    U = np.transpose(A)@Lb   # U = A'
    X = N@U         # X = (A'A)^(-1)*(A'x). Absolut translations refined
    # Join (slerped rotation + refined translation (LUM) in one pose T(4x4) 
    List_Poses_SLERP_LUM = []
    aux_0001 = np.array([0,0,0,1])
    for i in range(1,n):
        Pose = np.vstack(( np.hstack((lista_R_origem[i], X[3*(i-1):3*i])), aux_0001)) 
        List_Poses_SLERP_LUM.append(Pose)
    # Add identity pose as the first pose:
    List_Poses_SLERP_LUM.insert(0, np.identity(4))
    return List_Poses_SLERP_LUM


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
o3d.pipelines.registration.global_optimization(pose_graph, 
    o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(), 
    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), 
    option)


# 7 - RECOVER ABSOLUTE POSES FROM THE GRAPH
absolute_poses_g2o = [pose_graph.nodes[i].pose for i in range(n_clouds)]


# 8 - APPLY SLERP+LUM GLOBAL OPTIMIZATIONS ON RELATIVE POSES
absolute_poses_LUM  = reconstruir_Ts_para_origem_LUM(relative_poses_FGR_GICP)
absolute_poses_SLERP = reconstruir_Ts_para_origem_SLERP(relative_poses_FGR_GICP)
absolute_poses_SLERP_LUM = reconstruir_Ts_para_origem_SLERP_LUM(relative_poses_FGR_GICP)


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

'''
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
'''

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


# 13 - PAINT AND DRAW LINES
# LUM = GREEN | SLERP = RED | SLERP+LUM = BLUE | G2O = YELLOW | GROUNDTRUTH = WHITE
linhas_LUM.paint_uniform_color([1,0,0])
linhas_SLERP.paint_uniform_color([0,1,0])
linhas_SLERP_LUM.paint_uniform_color([0,0,1])
linhas_g2o.paint_uniform_color([1,1,0])
linhas_groundtruth.paint_uniform_color([0,0,0])

list_line_sets = [linhas_LUM,linhas_SLERP,linhas_SLERP_LUM,linhas_g2o,linhas_groundtruth]
draw_circuit_lines(list_line_sets)


'''
# SAVE ALL TRANSFORMATIONS
for i in range(n_clouds):
    np.savetxt(f"absolute_poses_LUM/NCLT/pose{i}.txt", absolute_poses_LUM[i], fmt="%.10f")
    np.savetxt(f"absolute_poses_SLERP/NCLT/pose{i}.txt", absolute_poses_SLERP[i], fmt="%.10f")
    np.savetxt(f"absolute_poses_SLERP_LUM/NCLT/pose{i}.txt", absolute_poses_SLERP_LUM[i], fmt="%.10f")
'''
