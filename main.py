import cv2
import numpy as np
import os
import imageio.v2 as imageio

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
    ct = CentroidTracker()
    (H, W) = (None, None)

    # Define a posição inicial das janelas
    x_pos_frame = 0
    x_pos_roi = 450

    rects = []
    target_size = (600, 600)
    for imagem in imagens[1:]:
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

        #Colocar Id nas fibras candidatas e guardar estrutura (histórico) com posição (X,Y) e Frame (imagem)

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

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame_display, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame_display, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # Exibe o resultado
        cv2.imshow('Frame', frame_display)
        cv2.moveWindow('Frame', x_pos_frame, 10)

        cv2.imshow('ROI', roi)
        cv2.moveWindow('ROI', x_pos_frame + W, 10)

        # cv2.imshow('mask', mask)
        # cv2.moveWindow('mask', x_pos_roi + W, 10)



        # Encerra o loop ao pressionar a tecla 'q'
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    diretorio_imagens = 'D:/mestrado/Material TCC Douglas/Imagens/CP09_6/img/'
    rastrear_ponto_em_imagens(diretorio_imagens)

