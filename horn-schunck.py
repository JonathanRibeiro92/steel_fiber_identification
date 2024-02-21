import os

import matplotlib.pyplot as plt
import numpy as np
import cv2

def plotFlow(u, v, displayImg, x, y):
    # Cia grid de coordenadas
    x_coords = np.arange(0, u.shape[1], x)
    y_coords = np.arange(0, u.shape[0], y)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Plota vetores do fluxo ótico
    plt.figure()
    plt.imshow(displayImg, cmap='gray')
    plt.quiver(x_grid, y_grid, u[::y, ::x], v[::y, ::x], color='red', angles='xy', scale_units='xy', scale=1)
    plt.show()


def computeDerivatives(im1, im2):
    fx = cv2.Sobel(im1, cv2.CV_64F, 1, 0, ksize=3) - cv2.Sobel(im2, cv2.CV_64F, 1, 0, ksize=3)
    fy = cv2.Sobel(im1, cv2.CV_64F, 0, 1, ksize=3) - cv2.Sobel(im2, cv2.CV_64F, 0, 1, ksize=3)
    ft = im2 - im1
    return fx, fy, ft

def HS(im1, im2, alpha, ite, displayFlow, displayImg):
    # Converte as imagens em escala de cinza
    if len(im1.shape) == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    if len(im2.shape) == 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

    im1 = np.float32(im1)
    im2 = np.float32(im2)

    im1 = cv2.GaussianBlur(im1, (0, 0), 1)
    im2 = cv2.GaussianBlur(im2, (0, 0), 1)

    # inicializa a velocidade dos vetores
    u = np.zeros_like(im1)
    v = np.zeros_like(im1)

    # cálculo das derivadas
    fx, fy, ft = computeDerivatives(im1, im2)

    # Define kernel para média local
    kernel_1 = np.array([[1 / 12, 1 / 6, 1 / 12],
                         [1 / 6, 0, 1 / 6],
                         [1 / 12, 1 / 6, 1 / 12]])

    # iterações
    for i in range(ite):
        # calcula as médias locais das velocidades dos vetores
        uAvg = cv2.filter2D(u, -1, kernel_1, borderType=cv2.BORDER_REFLECT)
        vAvg = cv2.filter2D(v, -1, kernel_1, borderType=cv2.BORDER_REFLECT)


        denom = alpha ** 2 + fx ** 2 + fy ** 2
        u = uAvg - (fx * ((fx * uAvg) + (fy * vAvg) + ft)) / denom
        v = vAvg - (fy * ((fx * uAvg) + (fy * vAvg) + ft)) / denom


    u[np.isnan(u)] = 0
    v[np.isnan(v)] = 0

    # apresenta o fluxo
    if displayFlow:
        plotFlow(u, v, displayImg, 5, 5)

if __name__ == '__main__':
    diretorio_imagens = 'D:/mestrado/Material TCC Douglas/Imagens/recortadas/CP06_6/img/'
    window_size = 3  # Tamanho da janela para o método Lucas-Kanade
    files = os.listdir(diretorio_imagens)
    im1 = cv2.imread(os.path.join(diretorio_imagens, files[0]), cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(os.path.join(diretorio_imagens, files[1]), cv2.IMREAD_GRAYSCALE)

    HS(im1, im2, alpha=15, ite=200, displayFlow=True, displayImg=im1)



