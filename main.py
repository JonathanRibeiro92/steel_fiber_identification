import cv2
import numpy as np
import os
from collections import OrderedDict
from pyimagesearch.centroidtracker import CentroidTracker
from utils.utils import showFrames, plotCube, cutImages, generate_pointsNpArray, generateDictIdFrames, plotFibers, \
    generateVolumes
from tqdm import tqdm


# Função principal de rastreamento

def rastrear_ponto_em_imagens(imagens, showVideo=False, maxDisappeared=20, minDistance=100):
    # função do openCV que retira o fundo das imagens
    object_detector = cv2.createBackgroundSubtractorMOG2()

    if not imagens:
        print("Nenhuma imagem encontrada no diretório.")
        return

    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker(maxDisappeared=maxDisappeared, minDistance=minDistance)
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
        # frame_gray = cv2.resize(frame_gray, target_size)
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
        if showVideo:
            showFrames(W, imagem, mask, objects.items(), target_size, x_pos_frame)

        # Encerra o loop ao pressionar a tecla 'q'
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return frames_history



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    maxDisappeared = 20
    minDistance = 60

    params = 'maxDissapeared{}_minDistance{}'.format(maxDisappeared,minDistance)

    diretorioImagens = 'D:/mestrado/Material TCC Douglas/Imagens/'
    diretorioNovo = 'D:/mestrado/Material TCC Douglas/Amostras/Cortes_6x6/'
    pbar = tqdm(os.listdir(diretorioNovo))
    for amostra in pbar:
        pbar.set_description("Processando a amostra %s" % amostra)
        diretorio_amostras = '{}{}'.format(diretorioNovo, amostra)
        diretorioReferencia = diretorioImagens + amostra +'/'
        nome_amostra = diretorio_amostras.split('/')[-1]
        cropped_images = cutImages(diretorio_amostras, diretorioReferencia)

        cropped_images = cropped_images[684:725]
        dictHistory = rastrear_ponto_em_imagens(cropped_images, maxDisappeared=maxDisappeared, minDistance=minDistance)

        pointsNpArray = generate_pointsNpArray(dictHistory)
        # Criação da figura e do subplot 3D
        plotCube(pointsNpArray, nomeAmostra=amostra, show=False, params=params)

        dictIdFrames = generateDictIdFrames(dictHistory)
        qtdCentroids = len(dictIdFrames.keys())
        print('{} centroids na amostra {}'.format(qtdCentroids, amostra))
        plotFibers(dictIdFrames, dictHistory, nome_amostra, params=params)

        generateVolumes(pointsNpArray, amostra, nome_amostra, params=params)
        print('Encerrado processamento da amostra {}'.format(amostra))
        break
    print('Encerrado processamento!!')
