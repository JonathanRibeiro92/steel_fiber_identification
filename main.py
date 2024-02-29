import cv2
import numpy as np
import os
import imageio.v2 as imageio
from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pyimagesearch.centroidtracker import CentroidTracker


# Função principal de rastreamento

def rastrear_ponto_em_imagens(diretorio_imagens):
    imagens = [imageio.imread(os.path.join(diretorio_imagens, img)) for img in os.listdir(diretorio_imagens)]

    #função do openCV que retira o fundo das imagens
    object_detector = cv2.createBackgroundSubtractorMOG2()

    if not imagens:
        print("Nenhuma imagem encontrada no diretório.")
        return

    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker(maxDisappeared=50)
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

        frame_display = cv2.resize(imagem.copy(), target_size)

        frame_display2 = cv2.resize(imagem.copy(), target_size)

        # Colocar Id nas fibras candidatas e guardar estrutura (histórico) com posição (X,Y) e Frame (imagem)
        frames_history[frameID] = objects.items()
        frameID += 1

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame_display, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame_display, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        listaTeste = list(objects.items())
        if len(listaTeste) > 100:
            (objectIDtest, centroid2) = listaTeste[100]
            text = "ID {}".format(objectIDtest)
            cv2.putText(mask, text, (centroid2[0] - 10, centroid2[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(mask, (centroid2[0], centroid2[1]), 4, (0, 255, 0), -1)

        # Exibe o resultado
        # cv2.imshow('Frame', frame_display)
        # cv2.moveWindow('Frame', x_pos_frame, 10)

        cv2.imshow('ROI', mask)
        cv2.moveWindow('ROI', x_pos_frame + W, 10)

        # cv2.imshow('frame_display2', frame_display2)
        # cv2.moveWindow('frame_display2', x_pos_frame + W, 10)



        # Encerra o loop ao pressionar a tecla 'q'
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return frames_history

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    diretorio_imagens = 'D:/mestrado/Material TCC Douglas/Imagens/recortadas/CP06_6/img/'
    dictHistory = rastrear_ponto_em_imagens(diretorio_imagens)
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

    # Criação da figura e do subplot 3D
    fig = plt.figure(figsize=(20, 15))
    # ax = fig.add_subplot(111, projection='3d')

    qtdSlices = np.array(range(len(dictHistory.keys())))

    xx, zz = np.meshgrid(range(pointsNpArray[:, 0].max() + 20), range(pointsNpArray[:, 1].max() + 20))
    ax = plt.subplot(projection='3d')
    # ax.plot_surface(xx, yy, zz)
    # Adiciona planos ao longo do eixo Y cortando os valores de slice
    for value in np.unique(pointsNpArray[:, 2]):
        yy = value
        ax.plot_surface(xx, yy, zz, rstride=5, cstride=5,
                        color='darkblue', linewidth=0, alpha=0.2, antialiased=True, shade=True)

    ax.scatter3D(pointsNpArray[:, 0], pointsNpArray[:, 2], pointsNpArray[:, 1], c='r', marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Slice')
    ax.set_zlabel('Y')
    plt.show()

    print("qtd Frames {}".format(len(dictHistory.keys())))
    for (frameId, centroids) in dictHistory.items():
        print("Frame {}".format(frameId))
        print("qtd Centroids {}".format(len(centroids)))
        for i in range(len(centroids)):
            dictIdFrames[i] = []
        for value in centroids:
            id = value[0]
            pos = value[1]
            dictIdFrames[id].append(frameId)

    for (id, frames) in dictIdFrames.items():
        print("id {}".format(id))
        print("Frames {}".format(frames))
