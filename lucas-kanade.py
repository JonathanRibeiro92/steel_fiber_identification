import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def compute_derivatives(im1, im2):
    # Calcula derivada horizontal
    fx = np.convolve(im1.flatten(), 0.25 * np.array([-1, 1]), 'same').reshape(im1.shape) + \
         np.convolve(im2.flatten(), 0.25 * np.array([-1, 1]), 'same').reshape(im2.shape)

    # Calcula derivada vertical
    fy = np.convolve(im1.flatten(), 0.25 * np.array([-1, -1, 1, 1]), 'same').reshape(im1.shape) + \
         np.convolve(im2.flatten(), 0.25 * np.array([-1, -1, 1, 1]), 'same').reshape(im2.shape)

    # Calcula derivada temporal
    ft = np.convolve(im1.flatten(), 0.25 * np.ones(2), 'same').reshape(im1.shape) + \
         np.convolve(im2.flatten(), -0.25 * np.ones(2), 'same').reshape(im2.shape)

    return fx, fy, ft



def lucas_kanade(im1, im2, window_size):
    # Calcula as derivadas espaciais e temporais
    fx, fy, ft = compute_derivatives(im1, im2)

    u = np.zeros_like(im1)
    v = np.zeros_like(im2)

    half_window = window_size // 2

    for i in range(half_window, im1.shape[0] - half_window):
        for j in range(half_window, im1.shape[1] - half_window):
            # Extrai a janela de pixels na vizinhança
            cur_fx = fx[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1].flatten()
            cur_fy = fy[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1].flatten()
            cur_ft = -ft[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1].flatten()

            A = np.vstack((cur_fx, cur_fy)).T

            # Calcula os componentes de movimento
            U = np.linalg.lstsq(np.dot(A.T, A), np.dot(A.T, cur_ft), rcond=None)[0]

            u[i, j] = U[0]
            v[i, j] = U[1]

    return u, v


def plot_motion_vectors(u, v):
    y, x = np.mgrid[0:u.shape[0], 0:u.shape[1]]
    plt.figure(figsize=(10, 6))
    plt.imshow(np.zeros_like(u), cmap='gray')
    plt.quiver(x, y, u, v, color='r', angles='xy', scale_units='xy', scale=1)
    plt.title('Motion Vectors')
    plt.xlabel('Horizontal')
    plt.ylabel('Vertical')
    plt.gca().invert_yaxis()
    plt.show()


def lucas_kanade_on_images_in_directory(directory, window_size):
    # Obtém a lista de arquivos no diretório
    files = os.listdir(directory)

    # Inicializa os vetores de movimento
    u = np.zeros((1, 1))
    v = np.zeros((1, 1))

    for file_idx, file in enumerate(files[:-1]):  # Itera até o penúltimo arquivo
        # Verifica se os arquivos são imagens
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Carrega as imagens
            im1 = cv2.imread(os.path.join(directory, file), cv2.IMREAD_GRAYSCALE)
            im2 = cv2.imread(os.path.join(directory, files[file_idx + 1]), cv2.IMREAD_GRAYSCALE)
            if im1 is not None and im2 is not None:
                # Calcula os componentes de movimento para o par de imagens
                cur_u, cur_v = lucas_kanade(im1, im2, window_size)

                # Concatena os resultados aos vetores de movimento
                if u.shape == (1, 1):
                    u = cur_u
                    v = cur_v
                else:
                    u = np.concatenate((u, cur_u), axis=0)
                    v = np.concatenate((v, cur_v), axis=0)

    # Exibe os vetores de movimento
    plot_motion_vectors(u, v)



if __name__ == '__main__':
    diretorio_imagens = 'D:/mestrado/Material TCC Douglas/Imagens/CP06_6/img/'
    window_size = 3  # Tamanho da janela para o método Lucas-Kanade
    files = os.listdir(diretorio_imagens)
    im1 = cv2.imread(os.path.join(diretorio_imagens, files[0]), cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(os.path.join(diretorio_imagens, files[1]), cv2.IMREAD_GRAYSCALE)
    u, v = lucas_kanade(im1, im2, window_size)
    plot_motion_vectors(u, v)

    # lucas_kanade_on_images_in_directory(diretorio_imagens, window_size)