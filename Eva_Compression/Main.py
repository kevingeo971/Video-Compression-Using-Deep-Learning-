
# imports
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.utils.data as utils
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import scale
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
import pandas as pd


%load_ext autoreload
%autoreload 2

%pylab inline

import os
import cv2
import time

import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import imageio
import torchvision.datasets as dset

from sklearn.cluster import AgglomerativeClustering



