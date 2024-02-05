import cv2
import numpy as np
import os
import imageio.v2 as imageio

# Função principal de rastreamento usando Lucas-Kanade

def rastrear_ponto_em_imagens(diretorio_imagens):
    imagens = [imageio.imread(os.path.join(diretorio_imagens, img)) for img in os.listdir(diretorio_imagens)]

    object_detector = cv2.createBackgroundSubtractorMOG2()

    if not imagens:
        print("Nenhuma imagem encontrada no diretório.")
        return

    # Define a posição inicial das janelas
    x_pos_frame = 0
    x_pos_roi = 450

    for imagem in imagens[1:]:
        # Converte a imagem para escala de cinza
        frame_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        height, width = frame_gray.shape
        roi = frame_gray[:, :]
        mask = object_detector.apply(roi)

        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # cv2.drawContours(frame_gray, [cnt], -1, (0, 255, 0), 2)

        frame_display = imagem.copy()

        # Exibe o resultado
        cv2.imshow('Frame', frame_display)
        cv2.moveWindow('Frame', x_pos_frame, 10)

        cv2.imshow('ROI', roi)
        cv2.moveWindow('ROI', x_pos_frame + width, 10)

        cv2.imshow('mask', mask)
        cv2.moveWindow('mask', x_pos_roi + width, 10)



        # Encerra o loop ao pressionar a tecla 'q'
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    diretorio_imagens = 'D:/mestrado/Material TCC Douglas/Imagens/CP06_6/img/'
    rastrear_ponto_em_imagens(diretorio_imagens)

