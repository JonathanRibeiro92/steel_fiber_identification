import caffe
import numpy as np

# Caminho para o arquivo de definição do modelo (prototxt)
model_def = 'caminho/para/seu/modelo.prototxt'
# Caminho para o arquivo de pesos do modelo (caffemodel)
model_weights = 'caminho/para/seus/pesos.caffemodel'

# Carregar o modelo e os pesos
net = caffe.Net(model_def, model_weights, caffe.TEST)

def main():
    input_folder = "D:\\mestrado\\steel-fiber\\Image2LMDB\\img\\train"
    output_lmdb_path = "D:\\mestrado\\steel-fiber\\Image2LMDB\\output"
    convert_images_to_lmdb(input_folder, output_lmdb_path)

if __name__ == "__main__":
    main()