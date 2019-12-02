
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



parser = argparse.ArgumentParser(description='Arguments for Eva Storage')
parser.add_argument('-train',action='store_true',default=False,dest = 'train',
                    help='''Do you want to train your own network?
                    Default is False''')
parser.add_argument('-DETRAC',action='store_true',default=False,dest ='DETRAC',
                    help='Use UE-DETRAC Dataset. Default is False')
parser.add_argument('-path',action='store',required = True,dest ='path',
                    help='Add path to folder')



args = parser.parse_args()


print(args)
