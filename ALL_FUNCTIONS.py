import numpy as np
import copy
import time
import open3d as o3d
import quaternion as quat
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


# Load and clouds, pre-process and paint
# Path pasta onde estao as nuvens, only clouds there!
def carregar_nuvens_e_pre_processar(voxel_size,knn,std,paths):
    lista_nuvens = []
    with os.scandir(path) as entries:
        for entry in entries:
            caminho = path + "/" + entry.name
            nuvem = o3d.io.read_point_cloud(caminho)
            nuvem_amostrada = nuvem.voxel_down_sample(voxel_size=voxel_size)
            nuvem_filtrada, _ = nuvem_amostrada.remove_statistical_outlier(nb_neighbors=knn,std_ratio=std)
            cor = np.random.vonmises(0, 1, size=(1,3) )
            nuvem_filtrada.paint_uniform_color(cor)
            lista_nuvens.append(nuvem)
    return lista_nuvens


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


# Function to draw the set of axis and draw a line until close the loop.
# INPUT: list of absolute poses of the n clouds, the first should be the identity.
# Tamanho = the size of the axis (blue-Z, red-X, yellow-Y). 
# Nuvens =  the list of n clouds to draw together. OUTPUT: nothing, only draw.
def criar_linhas_e_frames_3D_em_poses(poses, tamanho):
    # Ler translacoes e rotacoes das poses e stackear tudo verticalmente
    matriz_translacoes = np.zeros((1,3))
    matriz_rotacoes = np.identity(3)
    for i in range(len(poses)-1):
        t = np.vstack((matriz_translacoes, np.transpose(np.transpose([poses[i][:3,3]]))))
        R = np.vstack((matriz_rotacoes, poses[i][:3,:3]))
        matriz_translacoes = t
        matriz_rotacoes = R
    # Inicializar lista das linhas e dos eixos:
    eixos, linhas = [], []
    for i in range(len(matriz_translacoes)):
        # Definir pares de pontos que definem as linhas:
        if i < len(matriz_translacoes)-1:
            linha = [i,i+1]
            linhas.append(linha)
        elif i == len(matriz_translacoes)-1:
            linha = [i,0]
            linhas.append(linha)
        # Montar frames 3D coloridos (eixos):
        t = matriz_translacoes[i,:]        # translacao do eixo
        R = matriz_rotacoes[i*3:3*(i+1),:] # rotacao do eixo
        # criar eixo na coordenada t (translacao da pose) e rotacionar:
        eixo = o3d.geometry.TriangleMesh.create_coordinate_frame(size=tamanho, origin=t)
        eixo.rotate(R,center=t)
        eixos.append(eixo)
    # montar conjunto de linhas (uma funcao junta todas numa so geometria):
    conjunto_linhas = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(matriz_translacoes),
                                            lines=o3d.utility.Vector2iVector(linhas))
    # Inicializar visualizador e janela:
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # Os eixos sao adicionadas um-a-um na visualizacao, o primeiro define a origem
    # nuvens tambem sao adicionar assim
    for i in range(len(eixos)):
        vis.add_geometry(eixos[i])
        vis.poll_events()
    # O conjunto de linhas eh adicionado na visualizacao todo de uma vez, pois eh um so objeto:
    vis.add_geometry(conjunto_linhas)
    vis.run()
    vis.destroy_window()


# Function to transform a quaternion(4x1) and a translation t(3x1) 
# in a homogenous matrix T(4x4). 
# INPUT: quaternio + 3x1 array. OUTPUT: 4x4 numpy matrix 
def transformar_quaternio_em_4x4(quaternio,translacao):
    aux = np.array([0,0,0,1])
    rotacao3x4 = np.hstack((quat.as_rotation_matrix(quaternio), np.transpose([translacao])))
    T = np.vstack((rotacao3x4,aux))
    return T


# Function to invert a pose. INPUT: 4x4 matrix. OUTPUT: 4x4 matrix.
def Transformar_de_volta(T_4x4):
    R_inv = np.transpose(T_4x4[0:3,0:3])
    t_inv = np.transpose([-R_inv@T_4x4[0:3,3]])
    T_inv = np.vstack((np.hstack((R_inv,t_inv)), np.array([0,0,0,1])))
    return T_inv


# Function to interpolate between two homogeneous transformations T1 e T2.
# INPUT: Interval [0,1] of interpolation. OUTPUT. Interpolated homogeneous T
def interpolar_duas_T(T1,T2, t):
    # Media da translacao
    t1 = T1[0:3,3]*(1-t)
    t2 = T2[0:3,3]*t
    t_interpolada = t1+t2
    t_interpolada = [[t_interpolada[0]],[t_interpolada[1]],[t_interpolada[2]]]
    # transformar matrizes de rotacoes para quaternios 
    q1 = quat.from_rotation_matrix(T1[0:3,0:3])
    q2 = quat.from_rotation_matrix(T2[0:3,0:3])
    # Calcular rotacao media por SLERP em t
    q_interpolado = quat.quaternion_time_series.slerp(q1,q2,0,1,t)
    # Montar T com R(3x3) + t(3x1)
    t = t_interpolada*np.ones((3,1))
    rotacao3x4 = np.hstack((quat.as_rotation_matrix(q_interpolado), t))
    Pose_interpolada = np.vstack((rotacao3x4, np.array([0,0,0,1])))
    return Pose_interpolada


# Function to compose two relative poses. 
# INPUT: (T21,T10) homogeneous transformations.
# T21 = transformation which transform the system 2 to the reference system 1.
# T10 = transformation which transform the system 1 to the reference system 0.
# OUTPUT: T20 (transformation which transform the system 1 to the reference system 0).  
def compor_duas_poses(T21,T10):
    R21, t21 = T21[:3,:3], T21[:3,3]
    R10, t10 = T10[:3,:3], T10[:3,3]
    R20, t20 = R21@R10, R10@t21 + t10
    T20 = np.vstack((np.hstack((R20, np.reshape(t20,(3,1)))), np.array([0,0,0,1])))
    return T20


# Function to draw registration result and preserv the original cloud.
# Both temporary clouds will be painted with random colors.
def desenhar_resultado_registro(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0.85,0.65,0.05])
    target_temp.paint_uniform_color([0.60,0.10,0.85])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


# Function to project a list of clouds in the XY plane:
def planificar_nuvens_em_xy(lista_nuvens):
    lista_nuvens_planas = []
    for i in range(len(lista_nuvens)):
        xyz = np.asarray(lista_nuvens[i].points)
        xyz[:,2] = 0
        nuvem_plana = o3d.geometry.PointCloud()
        nuvem_plana.points = o3d.utility.Vector3dVector(xyz)
        lista_nuvens_planas.append(nuvem_plana)
    return lista_nuvens_planas


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
        use_absolute_scale = True,   # padrao: False
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


# INPUT: (source,target) = a pair of point clouds. iterations = max iterations. 
# max_corres_dist = max distance to look for correspondences. 
# initial_T = initial transformation between clouds.
# OUTPUT: Open3D object containing RMSE between correspondences, fitness (sobreposition),
# a list of correspondences and the 4x4 homogeneous transformation. 
def GICP_robusto(source, target, max_corres_dist, initial_T, iterations):
    # Constroi kd-trees e calcula normais e covariancias das nuvens
    kd_tree_normais = o3d.geometry.KDTreeSearchParamRadius(radius = 0.20)
    source.estimate_normals(kd_tree_normais)
    target.estimate_normals(kd_tree_normais)
    source.estimate_covariances()
    target.estimate_covariances()
    print(f"Applying robust GICP")
    loss = o3d.pipelines.registration.GMLoss(k = 1.0) # standard deviation of the noise
    result_ICP = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_corres_dist,
        initial_T,  # initial transformation
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(loss),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = iterations))
    return result_ICP


# Multiscale generator using random sampling to simulate voxel_downsampling
# INPUT: nuvem to be generated scales; n_escalas = number of scales to generate; 
# voxel_inicial = voxel of the most sparce scale.
# OUTPUT: list of clouds in coarse to fine, one by scale.
# TODO Add this to the multi-scale fuction to acelerate multi-scale process
def amostragem_multiescala_otimizada(nuvem, n_escalas, voxel_inicial):
    nuvem_amostrada_inicial = nuvem.voxel_down_sample(voxel_inicial)
    total_pts = len(np.asarray(nuvem.points))
    pts_inici = len(np.asarray(nuvem_amostrada_inicial.points))
    #escalas = np.asarray([voxel_inicial*2**i for i in range(n_escalas)]) # cresce quadrado
    escalas = np.asarray([voxel_inicial+voxel_inicial*i for i in range(n_escalas)]) # cresce linear
    # Predizer procentagens
    a, b = 1.18397758, 5.09388767
    porcentagens = a*np.exp(-b*escalas)
    # Escalonar % em realacao a quantidade de pts. da nuvem amostrada por voxel_inicial
    porcentagens_escalonadas = porcentagens*total_pts/pts_inici
    porcentagens_normalizadas = porcentagens_escalonadas/np.linalg.norm(porcentagens_escalonadas)
    porcentagens_normalizadas = porcentagens_normalizadas[1:10]
    lista_nuvens_amostradas = []
    for i in range(n_escalas-1):
        nuvem_amostrada = nuvem_amostrada_inicial.random_down_sample(porcentagens_normalizadas[i])
        lista_nuvens_amostradas.append(nuvem_amostrada)
    lista_nuvens_amostradas.insert(0, nuvem_amostrada_inicial)
    lista_nuvens_amostradas = list(reversed(lista_nuvens_amostradas))
    return lista_nuvens_amostradas


# Function to create scales of downsampling voxels doubling them.
# INPUT: integer number of desired scales.
# OUTPUT: python list with the number of scales.
def create_scales(n_scales):
    voxel_radius = [0.1]
    for i in range(n_scales-1):
        voxel_radius.append(voxel_radius[-1]+voxel_radius[-1])
    return voxel_radius


# Multiscale GICP with variable number of scales
# INPUT: source, target = pair of point clouds to be registered 
# n_scales = how much scales to use in the registration.
# OUTPUT: registration_result = Open3D object with correspondence set, RMSE, 
# fitness and a 4x4 transformation.
def Multiscale_GICP(source, target, n_scales, itera_escala, T_ini):
    # Create voxel sizes for downsampling
    voxel_sizes = create_scales(n_scales)
    voxel_sizes.reverse()
    # Calculate max_correspondences_distances using radios form point cloud pair
    max_correspondence_distance = radius_from_cloud_pair(source, target)
    max_correspondence_distances = [max_correspondence_distance*(2**(-i)) for i in range(n_scales)] 
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
        print(f"Amostrando...")
        source_down = source_temp.voxel_down_sample(voxel_sizes[i])
        target_down = target_temp.voxel_down_sample(voxel_sizes[i])
        # Filtrar outliers
        print(f"Filtrando outliers...")
        source_clean, _ = source_down.remove_statistical_outlier(knn_filtro, std_filtro)
        target_clean, _ = target_down.remove_statistical_outlier(knn_filtro, std_filtro)
        # Estimar normais
        print(f"Estimando normais...")
        source_clean.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
        target_clean.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
        # Aplicar GICP
        result_icp = o3d.pipelines.registration.registration_generalized_icp(source_clean,
            target_clean,
            max_correspondence_distances[i],
            current_transformation,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(loss),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=itera_escala))
        current_transformation = result_icp.transformation
    return result_icp


# Do a coarse-to-fine registration using FGR and Multiscale GICP. 
def Coarse_to_fine_FGR_M_GICP(source, target, voxel_size):
    # FGR coarse registration
    result_FGR = registro_FGR(source, target, voxel_size)
    # Define M-GICP parameters
    n_scales = 3
    itera_escala = 100
    T_ini = result_FGR.transformation
    # Refine 
    result_M_GICP = Multiscale_GICP(source, target, n_scales, itera_escala, T_ini)
    # Get information matrix
    information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source,
        target, 
        voxel_size, 
        result_M_GICP.transformation)
    return result_M_GICP, information_matrix 


# Function that builds the pose-graph of clouds with pairwise registration. The value of k 
# is the number of registrtions that will be made with one cloud, if k=3, then each cloud is 
# registered in the next 3 clouds. Given n vertices (clouds) this will define a graph with 
# k(2n-k-1)/2 edges (relative pairwise transformations).
# INPUT: lista_nuvens = list of clouds; voxel_size = voxel used to downsample clouds in the list
# k = degree of connectivity; if k = n, then it will build a complete graph.
# OUTPUT: posegraph, a .json archive to be read with Open3D fuction "read_pose_graph".
def full_registration(lista_nuvens, voxel_size, k):
    print(f"\nPara n={len(lista_nuvens)} | k={k} serao feitos {k*(len(lista_nuvens)-k)+(k**2-k)/2} registros em pares\n")
    ok = 0
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(lista_nuvens)
    for source_id in range(n_pcds):
        for target_id in range(source_id+1, n_pcds):
            if (target_id == source_id+1):
                print("Odometric case:")
                print(f"Registering cloud {source_id} in cloud {target_id}")
                T_FGR_ICP, information_matrix = Coarse_to_fine_FGR_M_GICP(lista_nuvens[source_id],
                    lista_nuvens[target_id],
                    voxel_size)
                odometry = np.dot(T_FGR_ICP.transformation, odometry)
                # Salva no vertice do grafo a multiplicação anterior invertida, isto eh, 
                # salva a pose absoluta. Poses absolutas registram a nuvem n na nuvem 0.
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                # Cria a primeira aresta do grafo, onde eh salvo os id dos vertices 0->1, a transformacao,
                # a matriz de informacao, e o tipo de aresta: odometria ou de loopclosure.
                # Nas arestas estao as poses relativas, que registram a nuvem anterior na seguinte                    
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                                target_id,
                                                                                T_FGR_ICP.transformation,
                                                                                information_matrix,
                                                                                uncertain=False)) # False significa que a pose adveio do registro com a nuvem imediatamente posterior
                if T_FGR_ICP.fitness > 0.40:
                    print("Sucesso\n")
                    ok = ok+1
                else:
                    print("Falhou\n")
            elif (target_id != source_id+1) and (target_id-source_id <= k):
                print("Caso loopclosure:")
                print(f"Registro da nuvem {source_id} na nuvem {target_id}")
                T_FGR_ICP, information_matrix = Coarse_to_fine_FGR_M_GICP(lista_nuvens[source_id],
                    lista_nuvens[target_id],
                    voxel_size)
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                                target_id,
                                                                                T_FGR_ICP.transformation,
                                                                                information_matrix,
                                                                                uncertain=True))
                if T_FGR_ICP.fitness > 0.40:
                    print("Sucesso\n")
                    ok = ok+1
                else:
                    print("Falhou\n")
                # uncertain = True significa que a nuvem nao foi registrada na imediatamente posterior (caso loop closure)
            elif (target_id != source_id+1) and (target_id-source_id > k):
                continue
    print(f"{ok} sucessos de {k*(len(lista_nuvens)-k)+(k**2-k)/2} registros em pares. Taxa: {ok/(k*(len(lista_nuvens)-k)+(k**2-k)/2)}")
    return pose_graph


def cortar_nuvem_manual():
    print("Corte manual de geometrias")
    print("1) Pressione 'Y' duas vezes para alinhar na direção do eixo-y")
    print("2) Pressione 'K' para travar a tela e mudar para o modo de seleção")
    print("3) Arraste para seleção retangular,")
    print("   ou use ctrl + botao esquerdo para selecao poligonal")
    print("4) Pressione 'C' para salvar uma geometria selecionada")
    print("5) Pressione 'F' para o modo livre de visualizacao")
    pcd = o3d.io.read_point_cloud("pc0.pcd")
    o3d.visualization.draw_geometries_with_editing([pcd])


def escolher_pontos(pcd):
    print("")
    print("1) Selecione pelo menos 3 correspondências usando [shift + left click]")
    print("   Pressione [shift + right click] para desfazer a selecao")
    print("2) Depois de clicar nos pontos pressione 'Q' para fechar a janela")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # pontos selecionados pelo usuário
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def registro_manual(source,target):
    print("Registro manual")
    print("Visualizacao antes do registro")
    desenhar_resultado_registro(source, target, np.identity(4))
    # Escolha pontos de duas nuvens e crie correspondencias
    picked_id_source = escolher_pontos(source)
    picked_id_target = escolher_pontos(target)
    # Minimo de 3 pontos
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target
    # Estimar transformacao grosseira usando correspondencias
    print("Calculando transformacao usando pontos escolhidos")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    T_inicial = p2p.compute_transformation(source,target,o3d.utility.Vector2iVector(corr))
    desenhar_resultado_registro(source, target, T_inicial)
    print("")
    return T_inicial


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


# Function to calculate closure error.
# (acumulate all transformations of a closed circuit)
# INPUT: T_circuito = list of relative poses: T10, T21, T32, ...
# T32 = transformation which transform system 3 to system 2.
# OUTPUT: closure error (ideally should by the identity).
def Calcular_Erro_LoopClosure(T_circuito):
    # Obter lista de rotacoes para a origem 
    lista_rotacoes_origem = []
    for i in range(len(T_circuito)):
        LoopClosure = np.identity(3)
        for j in range(len(T_circuito)-i-1,-1,-1):        
            LoopClosure = LoopClosure@T_circuito[j][:3,:3] # compoe j matrizes de rotacao
        lista_rotacoes_origem.append(LoopClosure) # lista de rotacoes compostas para origem
    lista_rotacoes_origem = list(reversed(lista_rotacoes_origem)) # inverte a lista, ultima eh a LoopClosure
    LoopClosure_Rotacao = lista_rotacoes_origem[len(T_circuito)-1]
    # Loop closure da translacao
    LoopClosure_Translacao = T_circuito[0][0:3,3]
    for i in range(len(T_circuito)-1):
        aux_Lb = lista_rotacoes_origem[i]@T_circuito[i+1][0:3,3] 
        LoopClosure_Translacao = LoopClosure_Translacao + aux_Lb
    # Calcular o erro no loop da rotacao (distancia entre a matriz LoopClosure e a identidade)
    ErroNoLoop = np.linalg.norm(LoopClosure_Rotacao-np.identity(3),'fro')
    # Montar POSE LOOP CLOSURE
    Pose_LoopClosure = np.hstack((LoopClosure_Rotacao,np.transpose([LoopClosure_Translacao])))
    print(f"POSE Closure error:\n{Pose_LoopClosure}") 
    print(f"Distancia (Frobenious) para a identidade:\n{ErroNoLoop}")
    return Pose_LoopClosure


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


# Function to animate the registration of clouds in a dataset.
# INPUT: Lista_nuvens = list of clouds. Lista_poses = list of absolute poses.
# n_frames = how much frames to use, use more to a smoth animation.
# OUTPUT: nothing, only draw.
def Reconstrucao_animada_uma_de_cada_vez(Lista_nuvens,Lista_poses,n_frames):
    N = n_frames # N de frames
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # Adicionar nuvens na visualizacao:
    for i in range(len(Lista_nuvens)):
        vis.add_geometry(Lista_nuvens[i])  # ! ADICIONAR A NUVEM 0 PRIMEIRO !
    # Interpolar entre poses e a identidade n vezes
    Todos_os_frames = []
    for i in range(1,len(Lista_poses)):
        N_frames_de_uma_pose = [interpolar_duas_T(np.identity(4),Lista_poses[i],(j+1)/N) for j in range(N)]
        # Tem-se Poses*N frames em Todos_os_frames, que eh uma lista de lista
        Todos_os_frames.append(N_frames_de_uma_pose)
    for i in range(len(Todos_os_frames)):
        for j in range(len(Todos_os_frames[i])):
            Lista_nuvens[i+1].transform(Todos_os_frames[i][j-1])
            vis.update_geometry(Lista_nuvens[i+1])
            vis.poll_events()
            vis.update_renderer()
            # Eh necessario transformar a nuvem i de volta:
            frame_inverso = Transformar_de_volta(Todos_os_frames[i][j])
            Lista_nuvens[i+1].transform(frame_inverso)
    vis.run()
    vis.destroy_window


# Function to animate FGR in a list of clouds:
# INPUT: Lista_nuvens = list of clouds. voxel_size = voxel used to downsample the clouds.
# OUTPUT: nothing, only draw.
def registro_FGR_animado(lista_nuvens, voxel_size):
    n_nuvens = len(lista_nuvens)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(lista_nuvens[0])  # ! ADICIONAR A NUVEM 0 PRIMEIRO !
    T_circuito = [np.identity(4)]
    ok = 0
    for i in range(1,n_nuvens):
        # Calcular pose relativa:
        T, _ = FRG_ICP_colorido_plano_multiescala(lista_nuvens[i], lista_nuvens[i-1], voxel_size)
        T_circuito.append(T.transformation)
        # Calcular pose absoluta:
        pose = compor_duas_poses(T_circuito[i],T_circuito[i-1])
        # Transformar nuvem i da lista
        lista_nuvens[i].transform(T.transformation)
        # Adicionar nuvem transformada
        vis.add_geometry(lista_nuvens[i])
        # Atualizar lista de poses
        T_circuito[i] = pose
        vis.poll_events()
        print(f"{i} -> {i-1}")
        if T.fitness > 0.5:
            print("Sucesso")
            ok = ok + 1
        else:
            print("falhou")
    print(f"Total de registros: {n_nuvens-1} Sucessos: {ok}")
    vis.run()
    vis.destroy_window()
    return T_circuito


# Function that calculates n_frames between the identity and each of the poses in List_poses, 
# returns a matrix of lines = 4*n_poses*n_frames and 4 columns. The first 4*n_poses lines are 
# the frames of the first interpolation. List_poses has a length equal to the number of clouds, 
# because the function that composes relative poses, returns the last pose as being the composition 
# of all, the closure-error pose.
def Calcular_Frames(Lista_poses,Lista_poses_2,n_frames):
    n_poses = len(Lista_poses)-1
    # Inicializar matriz que recebe todos os n frames interpolados de todas as k poses: 
    n_colunas = 4
    n_linhas = 4*(n_poses)*n_frames
    Todos_os_frames = np.zeros((n_linhas,n_colunas))
    # Loop que calcula em cada iteracao n_poses*n_frames:
    for i in range(n_frames):
        # Inicializar matriz de 4*n_poses*n_frames iniciais de cada pose:
        frames_poses = np.zeros((4*n_poses,n_colunas))
        # Loop das poses, calcula um frame interpolado para cada pose da lista:
        for j in range(1,n_poses+1):
            # Interpolar entre cada pose e a identidade i vezes (n_frames):
            frame_j = interpolar_duas_T(Lista_poses_2[j-1], Lista_poses[j-1], (i+1)/n_frames)
            # Salvar os frames de cada pose:
            frames_poses[4*j-4:4*j,:] = frame_j
        # Salvar frames de cada pose de 4*len(lista_nuvens-1) em 4*len(lista_nuvens-1)
        Todos_os_frames[4*(n_poses)*i:4*(n_poses)*(i+1),:] = frames_poses
    # Todos_os_frames calculados entre todas as poses. Eh uma matriz com: 
    # n_frames*n_nuvens-1 matrizes 4x4 estaqueadas verticalmente
    return Todos_os_frames


# Function to animate all clouds in a list to their respective poses.
# INPUT: Lista_nuvens = list of clouds; n_frames = number of frames; 
# frames_interpolados = previous interpolated frames (numpy array) using the above fuction.
# OUTPUT: nothing, only draw.
def Reconstrucao_animada_todas_de_uma_vez(Lista_nuvens,n_frames,frames_interpolados):
    n_nuvens = len(Lista_nuvens)
    # Inicializar visualizador e carregar nuvens:
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for nuvem in range(len(Lista_nuvens)):
        vis.add_geometry(Lista_nuvens[nuvem]) # Adicionar o nuvem 0 primeiro sempre pois ela define a origem
    # Interpolar n_frames entre a identidade e todas as poses:
    Todos_os_frames = frames_interpolados
    # Aplicar frames nas nuvens, a funcao "transform" aplica os primeiros frames interpolados. 
    # Antes de ir para o proximo frame, todas as nuvens sao atualizadas por transform:
    for i in range(n_frames):
        for j in range(1,n_nuvens):
            Lista_nuvens[j].transform(Todos_os_frames[(4*(n_nuvens-1)*i)+(4*j-4):(4*(n_nuvens-1)*i)+(4*j),:])    
            vis.update_geometry(Lista_nuvens[j])
        vis.poll_events()
        vis.update_renderer()
        # Como nao se pode aplicar transformacoes por cima de nuvens ja transformandas eh necessario 
        # transformar cada nuvem de volta para onde ela estava. Nao atualiza-se a visualizacao, pois a
        # transformacao de volta eh apenas para que o proximo frame seja aplicado na nuvem onde ela estava:
        for j in range(1,n_nuvens):
            frame_inverso = Transformar_de_volta(Todos_os_frames[(4*(n_nuvens-1)*i)+(4*j-4):(4*(n_nuvens-1)*i)+(4*j),:])
            Lista_nuvens[j].transform(frame_inverso)





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


# Function that use the absolute poses to obtein the relative poses
# INPUT: list of n absolut poses (the first must be the identity)
# OUTPUT: list of n-1 relative poses as follows:
# [T1: N0<-N1], [T2: N1<-N2], ..., [Tn: N(n-1)<-N(n)]
def poses_absolutas_para_relativas(poses_absolutas):
    n = len(poses_absolutas)
    # inverter todas as poses abs menos a ultima:
    absolutas_invertidas = [Transformar_de_volta(poses_absolutas[i]) for i in range(n-1)]
    # Compor poses absolutas com absolutas invertidas, mas com 1 de defasagem entre elas
    relativas = [compor_duas_poses(poses_absolutas[i+1], absolutas_invertidas[i]) for i in range(n-1)]
    return relativas


# Function to plot RMSE vs ITERATIONS of ICP,
# INPUT: ICP result (log_loss should be activated as True).
# OUTPUT: Draw a graphic
def plot_rmse_vs_iteracoes(registration_result):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 5))
    axes.set_title("Inlier RMSE vs Iteration")
    axes.plot(registration_result.loss_log["index"].numpy(), 
        registration_result.loss_log["inlier_rmse"].numpy())


# Function to plot RMSE vs ITERATIONS vs SCALE of Multiscale ICP,
# INPUT: ICP result (log_loss should be activated as True).
# OUTPUT: Draw a graphic
def plot_RMSE_vs_iteracoes_por_escala(registration_result):
    scales = registration_result.loss_log["scale"].numpy()
    iterations = registration_result.loss_log["iteration"].numpy()
    num_scales = scales[-1][0] + 1
    fig, axes = plt.subplots(nrows=1, ncols=num_scales, figsize=(20, 5))
    masks = {}
    for scale in range(0, num_scales):
        masks[scale] = registration_result.loss_log["scale"] == scale
        rmse = registration_result.loss_log["inlier_rmse"][masks[scale]].numpy()
        iteration = registration_result.loss_log["iteration"][masks[scale]].numpy()
        title_prefix = "Scale Index: " + str(scale)
        axes[scale].set_title(title_prefix + " Inlier RMSE vs Iteration")
        axes[scale].plot(iteration, rmse)


# Plot the RMSE of pairs before and after with LINES
def plot_RMSE_line(RMSE_1, RMSE_2):
    rotulos = [(f"{i}-{i+1}") for i in range(len(RMSE_1)-1)]
    rotulos.append((f"{len(RMSE_1)-1}-0"))
    rotulos = rotulos + rotulos
    RMSE = RMSE_1 + RMSE_2
    registro_FGR = [(f"FGR") for i in range(len(RMSE_1))]
    registro_GICP = [(f"FGR+GICP") for i in range(len(RMSE_2))]
    registro = registro_FGR + registro_GICP
    df = pd.DataFrame({'Pairs': rotulos, 'RMSE': RMSE, 'Algorithm': registro})
    sns.lineplot(data = df, x='Pairs', y='RMSE', hue='Algorithm')
    plt.show()


# Plot the FITNESS of pairs before and after with LINES
def plot_fitness_line(fitness_1, fitness_2):
    rotulos = [(f"{i}-{i+1}") for i in range(len(fitness_1)-1)]
    rotulos.append((f"{len(fitness_1)-1}-0"))
    rotulos = rotulos + rotulos
    fitness = fitness_1 + fitness_2
    registro_FGR = [(f"FGR") for i in range(len(fitness_1))]
    registro_GICP = [(f"FGR+GICP") for i in range(len(fitness_2))]
    registro = registro_FGR + registro_GICP
    df = pd.DataFrame({'Pairs': rotulos, 'Fitness': fitness, 'Algorithm': registro})
    sns.lineplot(data = df, x='Pairs', y='Fitness', hue='Algorithm')
    plt.show()


# Bar plot of RMSE:
def plot_RMSE_BAR(RMSE_1,RMSE_2):
    rotulos = [(f"{i}-{i+1}") for i in range(len(RMSE_1)-1)]
    rotulos.append((f"{len(RMSE_1)-1}-0"))
    rotulos = rotulos + rotulos
    RMSE = RMSE_1 + RMSE_2
    registro_FGR = [(f"FGR") for i in range(len(RMSE_1))]
    registro_GICP = [(f"FGR+GICP") for i in range(len(RMSE_2))]
    registro = registro_FGR + registro_GICP
    df = pd.DataFrame({'Pairs': rotulos, 'RMSE': RMSE, 'Algorithm': registro})
    sns.barplot(data = df, x='Pairs', y='RMSE', hue='Algorithm')
    plt.show()


# Bar plot of Fitness:
def plot_Fitness_BAR(fitness_1,fitness_2):
    rotulos = [(f"{i}-{i+1}") for i in range(len(fitness_1)-1)]
    rotulos.append((f"{len(fitness_1)-1}-0"))
    rotulos = rotulos + rotulos
    fitness = fitness_1 + fitness_2
    registro_FGR = [(f"FGR") for i in range(len(fitness_1))]
    registro_GICP = [(f"FGR+GICP") for i in range(len(fitness_2))]
    registro = registro_FGR + registro_GICP
    df = pd.DataFrame({'Pairs': rotulos, 'Fitness': fitness, 'Algorithm': registro})
    sns.barplot(data = df, x='Pairs', y='Fitness', hue='Algorithm')
    plt.show()


def plot_bar_time(time_list):
    rotulos = [(f"{i}-{i+1}") for i in range(len(time_list)-1)]
    rotulos.append((f"{len(time_list)-1}-0"))
    d = {'pairs': rotulos, 'Time taken': time_list}
    sns.barplot(data = pd.DataFrame(d), x='pairs', y='Time taken')
    plt.show()


def plot_MSE_of_dataset(MSE_1, MSE_2, MSE_3, MSE_4, error_type):
    MSE_data = [MSE_1, MSE_2, MSE_3, MSE_4]
    error_type = [error_type, error_type, error_type, error_type]
    optimization = ["Original","LUM","SLERP","SLERP+LUM"]
    d = {'MSE (all poses)': MSE_data, 'Error': error_type, 'Optimization': optimization}
    sns.barplot(data = pd.DataFrame(d), x='Error', y='MSE (all poses)', hue='Optimization')
    plt.show()


# Fuction to generate a random rotation matrix.
def rand_rotation_matrix(deflection=1.0):
    randnums = np.random.uniform(size=(3,))
    theta, phi, z = randnums
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi                 # For direction of pole deflection.
    z = z * 2.0*deflection                # For magnitude of pole deflection.
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))
    st = np.sin(theta)
    ct = np.cos(theta)
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


# This fuction read 2 list of poses and subtract them. Return two lists 
# of distances: euclidean distances for pairs of translations; 
# Frobenious distances for pairs of rotations.
# INPUT: 2 lists with n poses each. OUTPUT: 2 lists of n distances (rot + trans).   
def subtract_squared_poses(list_poses_1, list_poses_2):
    # Check length
    if len(list_poses_1) != len(list_poses_2):
        raise Exception("The list of poses should be the same size")
    distances_R = []
    distances_t = []
    # Loop to subtract poses
    for i in range(len(list_poses_1)):
        d_poses = list_poses_1[i] - list_poses_2[i]
        d_squared = d_poses**2
        d_R = sum(sum(d_squared[:3,:3]))**(1/2)
        d_t = sum(d_squared[:3,3])**(1/2)
        distances_R.append(d_R)
        distances_t.append(d_t)
    return distances_R, distances_t


# Fuction to plot pose diferences in translations
def plot_antes_depois_translacoes(diff_t_1, diff_t_2):
    # Plot both differences in translations
    poses = [i for i in range(len(diff_t_1))]
    rotulos = [(f"{0}-{i+1}") for i in range(len(diff_t_1)-1)]
    rotulos.insert(0, "0-0")
    plt.xticks(poses, rotulos)
    plt.plot(poses, diff_t_1, label = "Translations before")
    plt.plot(poses, diff_t_2, label = "Translations with SLERP+LUM")
    # name the x,y axis
    plt.xlabel('Absolute poses')
    plt.ylabel('translation error (m)')
    # show the legend and grid
    plt.grid(True)
    plt.legend()
    plt.show()


# Fuction to plot pose diferences in rotations
def plot_antes_depois_rotacoes(diff_R_1, diff_R_2):
    # Plot both differences in rotations
    poses = [i for i in range(len(diff_R_1))]
    rotulos = [(f"{0}-{i+1}") for i in range(len(diff_R_1)-1)]
    rotulos.insert(0, "0-0")
    plt.xticks(poses, rotulos)
    plt.plot(poses, diff_R_1, label = "Rotations before")
    plt.plot(poses, diff_R_2, label = "Rotations with SLERP+LUM")
    # name the x,y axis
    plt.ylabel('rotation error [0, 1] (adim.)')
    plt.xlabel('Absolute poses')
    # show the legend and grid
    plt.grid(True)
    plt.legend()
    plt.show()


def colorir_voxels(pc0):
    # fit to unit cube
    pc0.scale(1 / np.max(pc0.get_max_bound() - pc0.get_min_bound()), center=pc0.get_center())
    # colorir
    cor = np.random.vonmises(0, 1, size=( len(pc0.points), 3))
    pc0.colors = o3d.utility.Vector3dVector(cor)
    # Make voxels
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pc0, voxel_size=0.1)
    o3d.visualization.draw_geometries([voxel_grid])
    o3d.visualization.draw([pc0])


# Extrac eigen value-based features of a point cloud (pc)
def extract_eigen_features(pc):
    # Centralizar
    centroid = pc.get_center()
    pc = np.asarray(pc.points) - centroid
    # Normalizar
    pc = pc / max(np.linalg.norm(np.amax(pc,0)), np.linalg.norm(np.amin(pc,0)))
    # Transformar de volta em PointCloud object
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc)
    # Calcular Matriz de Correlacao
    _, C = o3d.geometry.PointCloud.compute_mean_and_covariance(pc_o3d)
    # Decompor em autovalores e autovetores
    _,s,_ = np.linalg.svd(C)
    # Calcular eigen features
    # Calculo do somatorio eh feito antes, ele escapa do intervalo [0,1] se feito apos a normalizacao
    eig_sum = s[0]+s[1]+s[2]
    # Normalizar autovalores
    s = s/np.linalg.norm(s)
    lin = (s[0] - s[1]) / s[0]
    pla = (s[1] - s[2]) / s[0]
    esf = s[2] / s[0]
    cur = s[2] / (s[0]+s[1]+s[2])
    ani = s[0] - s[2] / s[0]
    omn = (s[0] * s[1] * s[2])**(1/3)
    eigen_vector = np.array([lin, pla, esf, cur, ani, omn, eig_sum])
    return eigen_vector


# Function to draw n correspondences between a pair of registered point clouds
def draw_correspondences(source, target, registration_result, n):
    # Select correspondence set
    point_pairs = np.asarray(registration_result.correspondence_set)
    # Select n random correspondences
    random = np.random.randint( len( np.asarray(registration_result.correspondence_set) ), size=n )
    random_point_pairs = point_pairs[random, :]
    # Make lines between correspondences
    line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(source, target, random_point_pairs)
    # Draw line set with clouds
    o3d.visualization.draw_geometries([source, target, line_set])


# Function to analyse knn distances of points inside a single cloud.
# Useful for density analyses.
def plot_cloud_knn_distances(pc1,pc2):
    dists_1 = pc1.compute_nearest_neighbor_distance()
    dists_2 = pc2.compute_nearest_neighbor_distance()
    # Transformar distancias calculadas em data frames para visualizar estatisticas 
    rotulos_1 = ['Voxel downsampling' for i in range(len(dists_1))]
    Distancias_1 = pd.DataFrame({'Knn distances': dists_1, 'class': rotulos_1})
    rotulos_2 = ['Hybrid downsampling' for i in range(len(dists_2))]
    Distancias_2 = pd.DataFrame({'Knn distances': dists_2, 'class': rotulos_2})    
    df = pd.concat([Distancias_1,Distancias_2], axis=0)
    print("Boxplot of knn distances from borh clouds")
    sns.boxplot(data=df, x="Knn distances", y="class")
    plt.show()


# Fuction to claculate mean radius from a pair of point clouds.
# OUTPUT: float number. INPUT: pair of point clouds. 
def radius_from_cloud_pair(source, target):
    xyz_max = source.get_max_bound()
    xyz_min = source.get_min_bound()
    dif_1 = xyz_max - xyz_min
    xyz_max = target.get_max_bound()
    xyz_min = target.get_min_bound()
    dif_2 = xyz_max - xyz_min
    rad_1 = (dif_1[0]*dif_1[1]*dif_1[2])**(1/3)
    rad_2 = (dif_2[0]*dif_2[1]*dif_2[2])**(1/3)
    return (rad_1+rad_2)/2


# Function to rotate a cloud by 1 degree around Z axis and animate.
def rotate_cloud(pcd):
    R_z = np.array([[0.9993908, -0.0348995,  0.0000000],
                    [0.0348995,  0.9993908,  0.0000000],
                    [0.0000000,  0.0000000,  1.0000000]])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    for i in range(3600):
        pcd.rotate(R_z)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()   
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
