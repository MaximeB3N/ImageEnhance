import cv2
from cv2 import dnn_superres
from tqdm import trange
from time import time
import Tool
# Create an SR object
def Enhance(name_path,ouput_name):
    
    sr = dnn_superres.DnnSuperResImpl_create()

    name_init = name_path
    name_final = ouput_name
    # Read image
    image = cv2.imread(name_init)
    if (max(image.shape[0],image.shape[1])<=1200):
        N = 1000
    elif (max(image.shape[0],image.shape[1])/3+1<=1200):
        N = max(image.shape[0],image.shape[1])/3+1
    else:
        N = 1000

        
    shape_init = image.shape
    # Read the desired model
    path = 'EDSR_x3.pb'
    sr.readModel(path)

    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("edsr", 3)

    #We create a list of small images that can be the model's inputs
    imagesList = Tool.cut_image(image,N)
    print("Quadrillage de l'image initiale [OK]")

    #We enhance every small images
    imagesList_enhance = []
    print("Amélioration de ",len(imagesList)*len(imagesList[0])," images de taille initiale",N,"x",N,"en cours...")
    for i in range(len(imagesList)):
        S=[]
        for j in trange(len(imagesList[0])):
            S.append(sr.upsample(imagesList[i][j]))
        imagesList_enhance.append(S)

    print("Amélioration de chacune des images [OK]")

    #We reconstruct the big image with the list of enhanced images

    final = Tool.reconstruct_image(imagesList_enhance, shape_init)
    print("Reconstruction de l'image améliorée [OK]")

    # Save the image
    cv2.imwrite("./"+name_final, final)
    print("Image enregristrée dans le dossier de ce fichier sous le nom "+name_final+" [OK]")
