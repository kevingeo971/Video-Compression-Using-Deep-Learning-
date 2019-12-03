
# imports
import cv2
import pickle
import os
import time
import imageio
import numpy as np
from tqdm import tqdm
import argparse
from glob import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision as tv
import torch.optim as optim
import torch.utils.data as utils
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as dset
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.transforms as transforms
from sklearn.preprocessing import scale
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from CAE import CAE
from cluster import ClusterModule
from load_data import load_data
from train import train


#parser for user input, need to know where the frames are and whether they are being used to train
#or need encoding
parser = argparse.ArgumentParser(description='Arguments for Eva Storage')
parser.add_argument('-train',action='store_true',default=False,dest = 'train',
                    help='''Do you want to train your own network?
                    Default is False''')
parser.add_argument('-DETRAC',action='store_true',default=False,dest ='DETRAC',
                    help='Use UE-DETRAC Dataset. Default is False')
parser.add_argument('-path',action='store',required = True,dest ='path',
                    help='Add path to folder')
args = parser.parse_args()

path = args.path
train = args.train
DETRAC = args.DETRAC

if train:
    pass
    #TO DO call training function 
else:
    test_loader = load_data(path,DETRAC)
    
    images, _ = next(iter(test_loader)) 
    print("Shape : ",images[7:8].shape)
    
    model_n = CAE()
    model_n.load_state_dict(torch.load("CAE_Full_data.pwf"))#using previsouly saved model 
    model_n.cuda()
    
    enc_frames = np.zeros((1,1250)) #shape placeholder

    for batch_idx, batch in enumerate(test_loader):
        output = model_n(batch[0].cuda(),encode=True) #encoding
        enc_frames = np.vstack((enc_frames, output.detach().cpu().numpy())) #stacking frame codecs
    enc_frames = enc_frames[1:,:]  #removing placeholder
    print(" Enc frame : ", enc_frames.shape) #shape check
    
    CM = ClusterModule()
    labels = CM.run(enc_frames)#cluster generation                     

    clusters_seen = []
    index_list = []

    #choosing first (temporal metric) member of each cluster group 
    for i in range(enc_frames.shape[0]):
        clust = labels[i]
        if clust not in clusters_seen:
            clusters_seen.append(clust)
            index_list.append(i)

    print(index_list) #returning index list
    #TO DO 
    #return list in text file and frames in compressed avi format
    #user input with save path


    