import numpy as np 
import open3d as o3d 
import time
import matplotlib.pyplot as plt


# Importar nuvem
nuvem = o3d.io.read_point_cloud("nuvens_amostradas_50_cm/originais/s12.pcd")


# Amostrar nuvem em 5 escalas decrescentes 
lista_escalas = [1.0, 0.8, 0.6, 0.4, 0.2] # metros
lista_nuvens_amostradas = []
for escala in range(len(lista_escalas)):
    nuvem_amostrada = nuvem.voxel_down_sample(voxel_size=lista_escalas[escala])
    lista_nuvens_amostradas.append(nuvem_amostrada)


# Filtrar cada nuvem com vizinhanca variavel
lista_knn_filtro = [10,20,30,40,50]
lista_std = [0.4, 0.8, 1.2, 1.6, 2.0]
lista_tempos_execucao  = []
lista_nuvens_filtradas = []
lista_porcentagens_filtradas = []
for i in range(5):
    for j in range(5):
        start = time.time()
        nuvem_filtrada, pts_filtrados = lista_nuvens_amostradas[i].remove_statistical_outlier(
            nb_neighbors=lista_knn_filtro[1],
            std_ratio=lista_std[j],  
            print_progress = True)   
        end = time.time()
        tempo = end - start
        porcentagem_filtrada = 1 - (len(pts_filtrados)/len(np.asarray(lista_nuvens_amostradas[i].points)))
        # Salvar valores calculados 
        lista_tempos_execucao.append(tempo)
        lista_nuvens_filtradas.append(nuvem_filtrada)
        lista_porcentagens_filtradas.append(porcentagem_filtrada)


# Lista de valores observados 
z = lista_porcentagens_filtradas
#z = lista_tempos_execucao

# Plot de barras em 3D 
fig = plt.figure()
ax = plt.axes(projection = "3d")

data = np.reshape(np.asarray(z)*100, (5,5))

numOfRows = 5
numOfCols = 5

xpos = np.arange(0, numOfCols, 1)
ypos = np.arange(0, numOfRows, 1)
xpos, ypos = np.meshgrid(xpos + 0.5, ypos + 0.5)
# Posicao cas colunas
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros(numOfCols * numOfRows)
# Lagura das colunas
dx = np.ones(numOfRows * numOfCols) * 0.1
dy = np.ones(numOfCols * numOfRows) * 0.1
dz = data.flatten()
# Passar argumentos pra funcao bar3d
s = ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
ax.set_xticklabels(lista_std)
ax.set_yticklabels(lista_escalas)

ax.set_xlabel('Desvios padrões')
ax.set_ylabel('Escalas de amostragem (voxel)')
ax.set_zlabel('% pts. filtrados (knn = 20)')
 
plt.show()
'''
# Transladar nuvens amostradas e filtradas com d.p. variavel e escala variavel para knn=10.
t_x = np.arange(0, 1500, 300)
t_y = np.arange(0, 1500, 300)
t_x, t_y = np.meshgrid(t_x, t_y)

pc_0 = lista_nuvens_filtradas[0]
pc_6 = lista_nuvens_filtradas[6].translate(t)
pc_12 = lista_nuvens_filtradas[12].translate(t*2)
pc_18 = lista_nuvens_filtradas[18].translate(t*3)
pc_24 = lista_nuvens_filtradas[24].translate(t*4)
lista = [pc_0,pc_6,pc_12,pc_18,pc_24]
# Mostrar nuvens
o3d.visualization.draw_geometries(lista)
'''

