import cv2
import threading
import numpy as np
from cv2 import dnn_superres
from tqdm import trange
from time import time
from pathlib import Path

import code.Tool as Tool


# Create an SR object
def Enhance(input_path, name_final, output_path, temp_path, path_model, n_jobs=1, model_type="edsr", scale=4):
    
    # Read image
    image = cv2.imread(input_path)

    if (max(image.shape[0],image.shape[1])/scale+1<=1200):
        N = max(image.shape[0],image.shape[1])/scale+1
    else:
        N = 800

        
    shape_init = image.shape

    #We create a list of small images that can be the model's inputs
    imagesList = Tool.cut_image(image,N)
    print("Quadrillage de l'image initiale [OK]")

    #We enhance every small images
    print("Amélioration de ",len(imagesList)*len(imagesList[0])," images de taille initiale",N,"x",N,"en cours...")

    Enhance_tile_parallel(imagesList, temp_path, path_model, n_jobs, model_type, scale)

    print("Amélioration de chacune des images [OK]")

    imagesList_enhance = load_enhanced_images(temp_path, imagesList)
    #We reconstruct the big image with the list of enhanced images

    final = Tool.reconstruct_image(imagesList_enhance, shape_init, scale)
    print("Reconstruction de l'image améliorée [OK]")

    # Save the image
    if name_final is None:
        name_final = input_path.split("/")[-1].split(".")[0] + "_enhanced.jpg"
    path_final = str(Path(output_path) / name_final)

    if path_final.endswith("jpg"):
        cv2.imwrite(path_final, final)
    else:
        cv2.imwrite(path_final + ".jpg", final)

    print("Image enregristrée dans le dossier de ce fichier sous le nom "+name_final+" [OK]")


def Enhance_tile(model, imagesList, temp_path, n_jobs, k):
    for i in range(len(imagesList)):
        for j in range(len(imagesList[0])):
            if (i + j*len(imagesList[0]))% n_jobs== k:
                final = model.upsample(imagesList[i][j])
                np.save(Path(temp_path) / (str(i) + "_" + str(j) + ".npy"), final)


def Enhance_tile_parallel(imagesList, temp_path, path_model, n_jobs, model_type, scale):

    models = [dnn_superres.DnnSuperResImpl_create() for _ in range(n_jobs)]

    for model in models:
        model.readModel(path_model)
        model.setModel(model_type, scale)

    threads = [threading.Thread(target=(lambda k: Enhance_tile(models[k], imagesList, temp_path, n_jobs, k)), args=(k,)) for k in range(n_jobs)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

def load_enhanced_images(temp_path, imagesList):

    S = []
    for i in range(len(imagesList)):
        S.append([])
        for j in range(len(imagesList[0])):
            S[i].append(np.load(Path(temp_path) / (str(i) + "_" + str(j) + ".npy")))

    return S
