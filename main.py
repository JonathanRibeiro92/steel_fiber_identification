import random

import cv2
import numpy as np
import os
import imageio.v2 as imageio
from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
import glob

from pyimagesearch.centroidtracker import CentroidTracker


# Função para gerar o mesh stl 3D
def generateStl(pointsNpArray):
    # Tamanho do paralelepípedo
    xx = pointsNpArray[:, 0].max() + 20
    yy = pointsNpArray[:, 2].max()
    zz = pointsNpArray[:, 1].max() + 20

    paralelepipedo_vertices = np.array([
        [0, 0, 0],
        [0, yy, 0],
        [xx, yy, 0],
        [xx, 0, 0],
        [0, 0, zz],
        [0, yy, zz],
        [xx, yy, zz],
        [xx, 0, zz]
    ])

    print("Dimensões de paralelepipedo_vertices:", paralelepipedo_vertices.shape)

    paralelepipedo_faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
        [4, 7, 6],
        [4, 6, 5]
    ])

    # Combine os vértices e faces das fibras com as do paralelepípedo
    all_vertices = np.vstack([paralelepipedo_vertices, pointsNpArray])
    all_faces = np.vstack([
        np.arange(len(paralelepipedo_vertices)),  # Índices dos vértices do paralelepípedo
        paralelepipedo_faces,  # Índices das faces do paralelepípedo
        np.arange(len(paralelepipedo_vertices), len(paralelepipedo_vertices) + len(pointsNpArray))  # Índices das fibras
    ])

    print("Dimensões de all_vertices:", all_vertices.shape)
    print("Dimensões de all_faces:", all_faces.shape)

    # Crie o objeto mesh
    combined_mesh = mesh.Mesh(np.zeros(len(all_faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(all_faces):
        for j in range(3):
            combined_mesh.vectors[i][j] = all_vertices[face[j]]

    combined_mesh.save('fibers_stl_example_01.stl')


# Função principal de rastreamento

def rastrear_ponto_em_imagens(imagens):

    # função do openCV que retira o fundo das imagens
    object_detector = cv2.createBackgroundSubtractorMOG2()

    if not imagens:
        print("Nenhuma imagem encontrada no diretório.")
        return

    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker(maxDisappeared=4)
    (H, W) = (None, None)

    # Define a posição inicial das janelas
    x_pos_frame = 0
    x_pos_roi = 450
    frames_history = OrderedDict()

    target_size = (600, 600)
    frameID = 0
    for imagem in imagens:
        # Converte a imagem para escala de cinza
        frame_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.resize(frame_gray, target_size)
        # if the frame dimensions are None, grab them
        if W is None or H is None:
            (H, W) = frame_gray.shape[:2]

        roi = frame_gray[:, :]
        mask = object_detector.apply(roi)

        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rects = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            box = np.array([x, y, x + w, y + h])
            rects.append(box.astype("int"))
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # cv2.drawContours(frame_gray, [cnt], -1, (0, 255, 0), 2)

        # update our centroid tracker using the computed set of bounding
        # box rectangles
        objects = ct.update(rects)

        # Colocar Id nas fibras candidatas e guardar estrutura (histórico) com posição (X,Y) e Frame (imagem)
        frames_history[frameID] = objects.copy().items()
        frameID += 1

        showFrames(W, imagem, mask, objects.items(), target_size, x_pos_frame)

        # Encerra o loop ao pressionar a tecla 'q'
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return frames_history


def showFrames(W, imagem, mask, centroidsDict, target_size, x_pos_frame):
    frame_display = cv2.resize(imagem.copy(), target_size)
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


def plotCube(pointsNpArray):
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
    plt.show()


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    diretorio_amostras = 'D:/mestrado/Material TCC Douglas/Imagens/CP07_6'

    cropped_images = cutImages(diretorio_amostras)
    dictHistory = rastrear_ponto_em_imagens(cropped_images)
    dictIdFrames = OrderedDict()

    points = []
    for (frameId, centroids) in dictHistory.items():
        for value in centroids:
            id = value[0]
            z = frameId
            pos = value[1]
            x, y = pos
            points.append([x, y, z])

    pointsNpArray = np.array(points)

    print("Dimensões de pointsNpArray:", pointsNpArray.shape)

    # Criação da figura e do subplot 3D
    plotCube(pointsNpArray)

    # Geração do STL
    # generateStl(pointsNpArray)

    print("qtd Frames {}".format(len(dictHistory.keys())))
    for (frameId, centroids) in dictHistory.items():
        for value in centroids:
            id = value[0]
            pos = value[1]
            if dictIdFrames.keys().__contains__(id):
                dictIdFrames[id].append(frameId)
            else:
                dictIdFrames[id] = [frameId]

    for (id, frames) in dictIdFrames.items():
        print("id {}".format(id))
        print("Frames {}".format(frames))

    fiberExample = []
    centroidId = random.randint(0, len(dictIdFrames.keys()))
    for frameId in dictIdFrames[centroidId]:
        for value in dictHistory[frameId]:
            if value[0] != centroidId:
                continue
            z = frameId
            pos = value[1]
            x, y = pos
            fiberExample.append([x, y, z])
    fiberExample = np.array(fiberExample)

    plotCube(fiberExample)
