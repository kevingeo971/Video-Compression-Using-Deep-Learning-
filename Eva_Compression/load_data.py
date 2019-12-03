from glob import glob
from tqdm import tqdm
import imageio
import numpy as np
import cv2
import torch 
import torch.utils as utils
import os
import torch.utils.data as utils

def load_data(path, detrac = False):

    data = []

    if detrac:
        c = 0
        for i in tqdm(os.listdir('DETRAC-Images/')):
            n_data = []
            c+=1
            for j in range(1, len(os.listdir('DETRAC-Images/' + i + '/'))+1):
                j = str(j)
                jj= 5-len(j)
                k = "img" + jj*"0" +j +".jpg"
                img = imageio.imread('DETRAC-Images/' + i + '/' + k + '/')
                #print("image", img)
                n_data.append(cv2.resize(img, (100, 100), interpolation = cv2.INTER_AREA))
            n_data = np.array(n_data)
            print(n_data.shape)                          
            data = n_data.copy()
            #data.append(n_data)
            #if c ==1:
            break
        #data = np.squeeze(np.array(data))
    else:
        frames = glob(path + '/*.jpg')
        
        for f in tqdm(frames):
            img = imageio.imread(f)
            data.append(cv2.resize(img, (100, 100), interpolation = cv2.INTER_AREA))
                         
    data_load = np.array(data).transpose(0,3,1,2)
    print(data_load.shape)
    data_load= (data_load - 255) / 255
    #data_load = data_load/255

    batch_size = 8

    tensor_x = torch.stack([torch.Tensor(i) for i in data_load])

    train_dataset = utils.TensorDataset(tensor_x,tensor_x)
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader