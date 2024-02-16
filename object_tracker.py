# import the necessary packages
import os

import cv2
import imageio as iio
import imutils as imutils
import numpy as np

from pyimagesearch.centroidtracker import CentroidTracker

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
#                 help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
#                 help="path to Caffe pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#                 help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())


def preprocess_image(image, target_size):
    # Redimensionar a imagem
    resized_image = cv2.resize(image, target_size)

    # Normalizar os valores dos pixels
    normalized_image = resized_image.astype(np.float32) / 255.0  # Normalização para o intervalo [0, 1]

    # Se necessário, subtrair a média e dividir pelo desvio padrão
    # normalized_image = (normalized_image - mean) / std

    return normalized_image

def rastrear_ponto_cnn(diretorio_imagens):
    imagens = [iio.v3.imread(os.path.join(diretorio_imagens, img)) for img in os.listdir(diretorio_imagens)]

    if not imagens:
        print("Nenhuma imagem encontrada no diretório.")
        return

    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker()
    (H, W) = (None, None)

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')


    # loop over the frames from the video stream
    for imagem in imagens[1:]:

        frame = preprocess_image(imagem, (400,400))

        # if the frame dimensions are None, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the frame, pass it through the network,
        # obtain our output predictions, and initialize the list of
        # bounding box rectangles
        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
                                     (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        rects = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # filter out weak detections by ensuring the predicted
            # probability is greater than a minimum threshold
            if detections[0, 0, i, 2] > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object, then update the bounding box rectangles list
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                rects.append(box.astype("int"))

                # draw a bounding box surrounding the object so we can
                # visualize it
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)

        # update our centroid tracker using the computed set of bounding
        # box rectangles
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()


if __name__ == '__main__':
    diretorio_imagens = 'D:/mestrado/Material TCC Douglas/Imagens/CP09_6/img/'
    rastrear_ponto_cnn(diretorio_imagens)
