import glob
import os
from collections import OrderedDict

import cv2
import imageio
import numpy as np
from matplotlib import pyplot as plt

from Mesh3D.Mesh3D import create_volume_from_points, generate_mesh, generate_cube_mesh


def showFrames(W, imagem, mask, centroidsDict, target_size, x_pos_frame):
    # frame_display = cv2.resize(imagem.copy(), target_size)
    frame_display = imagem.copy()
    # loop over the tracked objects
    for (objectID, centroid) in centroidsDict:
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame_display, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame_display, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    # Exibe o resultado
    cv2.imshow('Frame', frame_display)
    cv2.moveWindow('Frame', x_pos_frame, 10)
    cv2.imshow('ROI', mask)
    cv2.moveWindow('ROI', x_pos_frame + W, 10)


def plotCube(pointsNpArray, centroidId=-1, nomeAmostra = None, show=True):
    fig = plt.figure(figsize=(20, 15))
    ax = plt.subplot(projection='3d')

    xx = pointsNpArray[:, 0].max() + 20
    yy = pointsNpArray[:, 2].max()
    zz = pointsNpArray[:, 1].max() + 20

    x = np.array([0, 0, xx, xx, 0, 0, xx, xx])
    y = np.array([0, yy, yy, 0, 0, yy, yy, 0])
    z = np.array([0, 0, 0, 0, zz, zz, zz, zz])
    ax.plot_surface(np.reshape(x, (2, 4)), np.reshape(y, (2, 4)), np.reshape(z, (2, 4)), alpha=0.5, color='darkblue')

    ax.scatter3D(pointsNpArray[:, 0], pointsNpArray[:, 2], pointsNpArray[:, 1], c='r', marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Slice')
    ax.set_zlabel('Y')
    nome_caminho = 'results/{}'.format(nomeAmostra)
    if centroidId!=-1 or nomeAmostra is not None:
        if not os.path.exists(nome_caminho):
            os.makedirs(nome_caminho)
        if centroidId != -1:
            nome_caminho = nome_caminho + '/fiber_example{}.png'.format(centroidId)
        else:
            nome_caminho = nome_caminho + '/fiber_total{}.png'
        plt.savefig(nome_caminho)
    if show:
        plt.show()
    plt.close(fig)



def find_blue_points(imagePath):
    # Carrega a imagem
    image = cv2.imread(imagePath)

    # Converte a imagem de BGR para HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Defina o intervalo de cor azul na escala HSV
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Cria uma máscara para os pixels que estão dentro do intervalo de cor azul
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Encontra os índices dos pixels azuis na imagem
    blue_indexes = np.where(mask == 255)

    # Obtém as coordenadas x e y dos pixels azuis
    blue_points = list(zip(blue_indexes[1], blue_indexes[0]))

    return blue_points


def cutImages(diretorio_amostras):
    diretorio_imagens = diretorio_amostras + '/img/'

    padrao_marker = 'Markers_Counter Window*'

    # Usa glob para encontrar arquivos que correspondam ao padrão
    arquivos_encontrados = glob.glob(os.path.join(diretorio_amostras, padrao_marker))
    mask_image_path = arquivos_encontrados[0]
    blue_points = find_blue_points(mask_image_path)

    min_x = min(coord[0] for coord in blue_points)
    max_x = max(coord[0] for coord in blue_points)
    min_y = min(coord[1] for coord in blue_points)
    max_y = max(coord[1] for coord in blue_points)

    cropped_images = []

    images = [imageio.imread(os.path.join(diretorio_imagens, img)) for img in os.listdir(diretorio_imagens)]

    for image in images:
        # Crop the region of the image delimited by the blue points
        cropped_image = image[min_y:max_y, min_x:max_x]
        image_with_rectangle = image.copy()
        cv2.rectangle(image_with_rectangle, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        cv2.imshow(f'Image with Rectangle', image_with_rectangle)
        cropped_images.append(cropped_image)
    return cropped_images

def generateVolumes(pointsNpArray, amostra, nome_amostra):
    mesh_from_points = generate_mesh(pointsNpArray)
    nome_caminho = 'results/{}'.format(amostra)
    # Salve o mesh em um arquivo STL
    mesh_from_points.save('{}/fibers_{}.stl'.format(nome_caminho, nome_amostra))
    volume = create_volume_from_points(pointsNpArray)
    cube_mesh = generate_cube_mesh(volume)
    cube_mesh.save('{}/mesh_cube_{}.stl'.format(nome_caminho, nome_amostra))


def generateDictIdFrames(dictHistory):
    dictIdFrames = OrderedDict()
    for (frameId, centroids) in dictHistory.items():
        for value in centroids:
            id = value[0]
            pos = value[1]
            if dictIdFrames.keys().__contains__(id):
                dictIdFrames[id].append(frameId)
            else:
                dictIdFrames[id] = [frameId]
    return dictIdFrames


def generate_pointsNpArray(dictHistory):
    points = []
    for (frameId, centroids) in dictHistory.items():
        for value in centroids:
            id = value[0]
            z = frameId
            pos = value[1]
            x, y = pos
            points.append([x, y, z])
    return np.array(points)

def plotFibers(dictIdFrames, dictHistory, nome_amostra):
    for centroidId in dictIdFrames.keys():
        fiberExample = []
        for frameId in dictIdFrames[centroidId]:
            for value in dictHistory[frameId]:
                if value[0] != centroidId:
                    continue
                z = frameId
                pos = value[1]
                x, y = pos
                fiberExample.append([x, y, z])
        fiberExample = np.array(fiberExample)

        plotCube(fiberExample, centroidId=centroidId, nomeAmostra=nome_amostra, show=False)