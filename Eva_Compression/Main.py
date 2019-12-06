
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
from train import train as additional_train
from write_original_video import write_original_video
from save_rep_video import save_compressed_video

# Argument Parser for Main.py:
"""
# Syntax: 
-train = boolean flag indicator model is to be to be retrained
-DETRAC = boolean flag indicator if using the UE DETRAC dataset 
-path = string path to data folder
-save_path = path to where 
"""

parser = argparse.ArgumentParser(description='Arguments for Eva Storage')

parser.add_argument('-train',action='store_true',default=False,dest = 'train',
                    help='''Do you want to train your own network?
                    Default is False''')

parser.add_argument('-DETRAC',action='store_true',default=False,dest ='DETRAC',
                    help='Use UE-DETRAC Dataset. Default is False')

parser.add_argument('-path',action='store',required = True,dest ='path',
                    help='Add path to data folder')

parser.add_argument('-save_original_path',action='store',required = False, default='',dest ='save_original_path',
                    help='Add path to save video. This is needed only if the original video needs to be scaled and saved')

parser.add_argument('-save_compressed_path',action='store',required = True, default='',dest ='save_compressed_path',
                    help='Add path to save Compressed video')

parser.add_argument('-scale',action='store', dest = 'scale', type=float,
                    help=''' The scale value or height width to save the original video. Needs to be between 0 and 1 ''')

parser.add_argument('-width',action='store', dest = 'width', type=int,
                    help=''' The scale value or height width to save the original video ''')

parser.add_argument('-height',action='store', dest = 'height',type=int,
                    help=''' The scale value or height width to save the original video ''')

parser.add_argument('-verbose',action='store_true',default=False,dest = 'verbose',
                    help=''' Display losses during training, It is false by default''')


args = parser.parse_args()

path = args.path
train = args.train
DETRAC = args.DETRAC
save_original_path = args.save_original_path
save_compressed_path = args.save_compressed_path
verbose = args.verbose
scale = args.scale
width = args.width
height = args.height

if path is not None:
    if scale is None and width is None:
        raise Exception("Scale or Height and Width Not passed along with save_original_path")
        

if scale is None and width is not None:
    scale = (width, height)
    


# Train/ Test condition. Train trains a new encoder module. Test uses
# a prexisting model trained on the UE-DETRAC Dataset

if train:    
    additional_train(path,DETRAC,train,2,verbose)
    print('\n Done\n\n')
    
else:
    
    # Dataloading
    test_loader, frame_list = load_data(path,DETRAC)
    images, _ = next(iter(test_loader)) 
    
    # Load pre-trained model 
    model_n = CAE()
    model_n.load_state_dict(torch.load("CAE_Full_data.pwf"))
    model_n.cuda()
    
    enc_frames = np.zeros((1,1250))

    print('\n Compressing Frames ..')
    
    for batch_idx, batch in enumerate(test_loader):
        output = model_n(batch[0].cuda(),encode=True)
        enc_frames = np.vstack((enc_frames, output.detach().cpu().numpy()))
    enc_frames = enc_frames[1:,:]  
    
    print('\n Clustering .. ')
    CM = ClusterModule()
    labels = CM.run(enc_frames)
    CM.plot_distribution()

    clusters_seen = []
    index_list = []
    for i in range(enc_frames.shape[0]):
        clust = labels[i]
        if clust not in clusters_seen:
            clusters_seen.append(clust)
            index_list.append(i)
    
    if scale is not None:
        print('\n Writing Original Scaled Video Data ..')    
        write_original_video(frame_list, scale = scale ,data_path = path, save_original_path = save_original_path,Detrac=DETRAC)
        
    
    save_compressed_video( frame_list, save_compressed_path , index_list)
    
    
    print( '\n Done\n ')


    