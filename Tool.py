import numpy as np

def cut_image(image,N):
    l,h,lr,hr = int(image.shape[0]/N),int(image.shape[1]/N),image.shape[0]%N,image.shape[1]%N
    Nl,Nh = l,h

    imagesList = []
    for i in range(0,Nl):
        S = []
        for j in range(0,Nh):
            S.append(np.copy(image[(i)*N:(i+1)*N,(j)*N:(j+1)*N,...]))
        
        imagesList.append(S)
    if lr!=0:
        S = []
        for j in range(0,Nh):
            S.append(np.copy(image[Nl*N:,(j)*N:(j+1)*N,...]))
        imagesList.append(S)
        
    if hr!=0:
        for i in range(0,len(imagesList)-1):
            imagesList[i].append(np.copy(image[(i)*N:(i+1)*N,Nh*N:,...]))
        if (len(imagesList)>0):
            imagesList[-1].append(np.copy(image[(len(imagesList)-1)*N:,Nh*N:,...]))
            
    return imagesList


def full_image(imagesList, image, ind_i, ind_j, i, j):
    for k in range(ind_i,ind_i+imagesList[i][j].shape[0]):
        for l in range(ind_j,ind_j+imagesList[i][j].shape[1]):
            for u in range(3):
                ia = int(imagesList[i][j][k-ind_i,l-ind_j,u])                 
                image[k,l,u] = ia

                
def reconstruct_image(imagesList, shape_init):
    l,h = shape_init[0]*3,shape_init[1]*3
    image = np.zeros((l,h,3),dtype = int)
    ind_i,ind_j = 0,0
    for i in range(len(imagesList)):
        ind_j = 0
        for j in range(len(imagesList[0])):
            full_image(imagesList,image,ind_i,ind_j,i,j)
            
            ind_j+=imagesList[i][j].shape[1]
            
        ind_i +=imagesList[i][j].shape[0]
    
    return image