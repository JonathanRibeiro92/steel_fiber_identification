import os
from typing import Dict, List
from collections import OrderedDict

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import blob_log

from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
from skimage.measure import label, regionprops, regionprops_table
from tqdm import tqdm

from utils.utils import cutImages, generate_pointsNpArray, plotCube, generate_pointsNpArrayFromBlobs, generateVolumes

square = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]])
def multi_dil(im, num, element=square):
    for i in range(num):
        im = dilation(im, element)
    return im

def multi_ero(im, num, element=square):
    for i in range(num):
        im = erosion(im, element)
    return im

def identificarPontosBlobs(imagem, threshold=0.1, bin_coef = 0.2 ) -> List:
  sample = imagem
  sample_g = rgb2gray(sample)
  sample_b = sample_g > bin_coef
  sample_c = multi_ero(multi_dil(sample_b,5),5)
  max_sigma=10
  blobs_log = blob_log(sample_c, max_sigma=max_sigma, num_sigma=10, threshold=threshold)
  # blobs_dog = blob_dog(sample_c, max_sigma=max_sigma, threshold=threshold)
  # blobs_doh = blob_doh(sample_c, max_sigma=max_sigma, threshold=threshold/10)
  coordenadaFibra = []
  listaFibrasCoordenadas = []
  dictFibrasCoordenadasIdentificadas = {}
  for blob in tqdm(blobs_log, desc="Blobs LoG"):
    y, x, area = blob
    coordenadaFibra.append(int(x))
    coordenadaFibra.append(int(y))
    listaFibrasCoordenadas.append(coordenadaFibra)
    coordenadaFibra = []
  dictFibrasCoordenadasIdentificadas['LoG'] = listaFibrasCoordenadas
  # listaFibrasCoordenadas = []

  # for blob in tqdm(blobs_dog, desc="Blobs DoG"):
  #   y, x, area = blob
  #   coordenadaFibra.append(int(x))
  #   coordenadaFibra.append(int(y))
  #   listaFibrasCoordenadas.append(coordenadaFibra)
  #   coordenadaFibra = []
  # dictFibrasCoordenadasIdentificadas['DoG'] = listaFibrasCoordenadas
  # listaFibrasCoordenadas = []
  #
  # for blob in tqdm(blobs_doh, desc="Blobs DoH"):
  #   y, x, area = blob
  #   coordenadaFibra.append(int(x))
  #   coordenadaFibra.append(int(y))
  #   listaFibrasCoordenadas.append(coordenadaFibra)
  #   coordenadaFibra = []
  # dictFibrasCoordenadasIdentificadas['DoH'] = listaFibrasCoordenadas

  # return dictFibrasCoordenadasIdentificadas
  return listaFibrasCoordenadas



if __name__ == '__main__':
    diretorioImagens = 'D:/mestrado/Material TCC Douglas/Imagens/'
    diretorioNovo = 'D:/mestrado/Material TCC Douglas/Amostras/Cortes_6x6/'
    threshold = 0.01
    bin_coef = 0.8
    pbar = tqdm(os.listdir(diretorioNovo))

    for amostra in pbar:
        pbar.set_description("Processando a amostra %s" % amostra)
        diretorio_amostras = '{}{}'.format(diretorioNovo, amostra)
        diretorioReferencia = diretorioImagens + amostra + '/'
        nome_amostra = diretorio_amostras.split('/')[-1]
        cropped_images = cutImages(diretorio_amostras, diretorioReferencia)
        frames_history = OrderedDict()
        frameId = 0
        listaImagens = cropped_images[684:725]
        for imagem in listaImagens:
            frames_history[frameId] = identificarPontosBlobs(imagem, threshold, bin_coef)
            frameId += 1


        pointsNpArray = generate_pointsNpArrayFromBlobs(frames_history)
        # Criação da figura e do subplot 3D
        plotCube(pointsNpArray, nomeAmostra=amostra, show=True)
        generateVolumes(pointsNpArray, amostra, nome_amostra)
        print("gerado volume da amostra %s" % amostra)
        break
