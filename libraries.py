import os
import cv2
import PIL
import glob
import torch
import scipy
import random
import shutil
import numpy as np
import pandas as pd
from math import exp
import torch.nn as nn
from PIL import Image as Img
from tqdm import tqdm
from scipy import ndimage
import torch.optim as optim
from sklearn import metrics
from torch.utils import data
from sklearn import datasets
from skimage.io import imsave
import matplotlib.pyplot as plt
import torch.nn.functional as F3
import torch.nn.functional as F9
from sklearn import linear_model
from skimage.feature import canny
from torch.autograd import Variable
import torchvision.models as models
from collections import OrderedDict
from torch.autograd import Variable
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt1
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from IPython.display import display, Image
from skimage.color import lab2rgb, rgb2lab
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.transforms.functional as F6
import torchvision.transforms.functional as F7
from sklearn.preprocessing import OneHotEncoder 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from sklearn.model_selection import cross_val_score, cross_val_predict