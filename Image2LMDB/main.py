import os
import lmdb
import numpy as np
import cv2
from tqdm import tqdm

def convert_images_to_lmdb(input_folder, output_lmdb_path):
    # Verifica se o caminho de saída existe, caso contrário, cria o diretório
    os.makedirs(output_lmdb_path, exist_ok=True)

    # Abre o banco de dados LMDB para escrita
    env = lmdb.open(output_lmdb_path, map_size=int(1e9))

    # Inicia uma transação
    with env.begin(write=True) as txn:
        # Loop pelas subpastas dentro da pasta principal
        for subfolder_name in os.listdir(input_folder):
            subfolder_path = os.path.join(input_folder, subfolder_name)
            # Verifica se é uma pasta
            if os.path.isdir(subfolder_path):
                # Inicializa a barra de progresso para esta subpasta
                progress_bar = tqdm(os.listdir(subfolder_path), desc=f"Processando {subfolder_name}", unit="imagem")
                # Loop pelas imagens na subpasta
                for img_name in progress_bar:
                    img_path = os.path.join(subfolder_path, img_name)
                    # Carrega a imagem usando o OpenCV
                    img = cv2.imread(img_path)
                    # Converte a imagem para array numpy
                    img_array = np.array(img)
                    # Codifica a imagem como bytes
                    img_bytes = img_array.tobytes()
                    # Insere a imagem no banco de dados LMDB
                    txn.put(img_name.encode(), img_bytes)
                    # Atualiza a barra de progresso
                    progress_bar.set_postfix({"Quantidade": txn.stat()["entries"]})
                # Fecha a barra de progresso
                progress_bar.close()

    # Fecha o banco de dados LMDB
    env.close()

def main():
    input_folder = "D:\\mestrado\\steel-fiber\\Image2LMDB\\img\\train"
    output_lmdb_path = "D:\\mestrado\\steel-fiber\\Image2LMDB\\output"
    convert_images_to_lmdb(input_folder, output_lmdb_path)

if __name__ == "__main__":
    main()
