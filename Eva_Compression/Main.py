
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
else:
    test_loader = load_data(path,DETRAC)
    
    images, _ = next(iter(test_loader)) 
    print("Shape : ",images[7:8].shape)
    
    model_n = CAE()
    model_n.load_state_dict(torch.load("CAE_Full_data.pwf"))
    model_n.cuda()
    
    enc_frames = np.zeros((1,1250))

    for batch_idx, batch in enumerate(test_loader):
        output = model_n(batch[0].cuda(),encode=True)
        enc_frames = np.vstack((enc_frames, output.detach().cpu().numpy()))
    enc_frames = enc_frames[1:,:]  
    print(" Enc frame : ", enc_frames.shape)
    
    CM = ClusterModule()
    labels = CM.run(enc_frames)                             

    clusters_seen = []
    index_list = []
    for i in range(enc_frames.shape[0]):
        clust = labels[i]
        if clust not in clusters_seen:
            clusters_seen.append(clust)
            index_list.append(i)

    print(index_list)


    