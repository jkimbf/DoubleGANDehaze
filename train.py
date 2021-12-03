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
from torch.utils.tensorboard import SummaryWriter


from Unet import UNet
from Discriminator import Discriminator
from Losses import *
from DU_Net import DU_Net
from DoubleGANNet import DoubleGANNet


class Dataset(torch.utils.data.Dataset):
    def __init__(self, haze_list, dehaze_list, augment=False):
        super().__init__()
        self.augment = augment
        self.haze_list = sorted(haze_list)
        self.dehaze_list = sorted(dehaze_list)
        
    def __len__(self):
        return 210

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.haze_list[index])
            item = self.load_item(0)
            

        return item


    def load_item(self, index):
        val = 1024*2           #crop size i.e hight and width
        size_data = 40         #depends on the no. of training images in the dataset
        height_data = 4657     #heigth of the training images
        width_data = 2833      #width of the training images

        numx = random.randint(0, height_data-val)
        numy = random.randint(0, width_data-val)
        # print(len(self.haze_list))
        # print(len(self.dehaze_list))
        haze_image = cv2.imread(self.haze_list[index%size_data])
        dehaze_image = cv2.imread(self.dehaze_list[index%size_data])
        haze_image = Img.fromarray(haze_image)
        dehaze_image = Img.fromarray(dehaze_image)

        haze_crop=haze_image.crop((numx, numy, numx+val, numy+val))
        dehaze_crop=dehaze_image.crop((numx, numy, numx+val, numy+val))
 
        haze_crop = haze_crop.resize((512,512), resample=PIL.Image.BICUBIC)
        dehaze_crop = dehaze_crop.resize((512,512), resample=PIL.Image.BICUBIC)

        haze_crop = np.array(haze_crop)
        dehaze_crop = np.array(dehaze_crop)
        haze_crop = cv2.cvtColor(haze_crop, cv2.COLOR_BGR2YCrCb)
        dehaze_crop = cv2.cvtColor(dehaze_crop, cv2.COLOR_BGR2YCrCb)
        haze_crop = self.to_tensor(haze_crop).cuda()
        dehaze_crop = self.to_tensor(dehaze_crop).cuda()
        
        return haze_crop.cuda(), dehaze_crop.cuda()
    
    def to_tensor(self, img):
        img_t = F.to_tensor(img).float()
        return img_t


    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


path_of_train_hazy_images = 'O-Haze/train/haze/*'
path_of_train_gt_images = 'O-Haze/train/gt/*'

images_paths_train_gt=glob.glob(path_of_train_gt_images)
image_paths_train_hazy=glob.glob(path_of_train_hazy_images)

train_dataset = Dataset(image_paths_train_hazy, images_paths_train_gt, augment=False)

train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=8,
            num_workers=0,
            drop_last=True,
            shuffle=False
        )

graph_gloss = []
input_unet_channel = 3
output_unet_channel = 3
input_dis_channel = 3
max_epochs = 100

""" ===================================Configurations=================================== """
# True: DoubleGANNet, False: BPPNet
using_ours = True
continue_training = True

# log_dir = "weights_211201"
log_dir = "ours_weights_211203"

starting_epoch = 100
ending_epoch = 300

dir_for_output_weights = 'ours_weights_211203'
""" ==================================================================================== """

if using_ours:
    # DUNet = DoubleGANNet(input_unet_channel, output_unet_channel, input_dis_channel).cuda()
    DUNet = DoubleGANNet(input_unet_channel, output_unet_channel, 7).cuda()
    print('Using DoubleGANNet architecture')
else:
    DUNet = DU_Net(input_unet_channel, output_unet_channel, input_dis_channel).cuda()
    print('Using BPPNet architecture')


if continue_training:
    print('Continue training from {} at epoch={}'.format(log_dir, starting_epoch))

    path_of_generator_weight = '{}/generator_{}.pth'.format(log_dir, starting_epoch)  #path where the weights of genertaor are stored
    path_of_discriminator_weight = '{}/discriminator_{}.pth'.format(log_dir, starting_epoch)  #path where the weights of discriminator are stored
    DUNet.load(path_of_generator_weight,path_of_discriminator_weight)


def train(start_epoch, end_epochs):
    # writer = SummaryWriter()
    for epoch in range(start_epoch, end_epochs):
        i=1
        mse_epoch = 0.0
        ssim_epoch = 0.0
        unet_epoch = 0.0
        for haze_images, dehaze_images, in train_loader:
            unet_loss, dis_loss, mse, ssim = DUNet.process(haze_images.cuda(), dehaze_images.cuda())
                
            DUNet.backward(unet_loss.cuda(), dis_loss.cuda())
            
            print('Epoch: '+str(epoch+1)+ ' || Batch: '+str(i)+ " || unet loss: "+str(unet_loss.cpu().item()) + " || dis loss: "+str(dis_loss.cpu().item()) + " || mse: "+str(mse.cpu().item()) + " | ssim:" + str(ssim.cpu().item()) )
            mse_epoch =  mse_epoch + mse.cpu().item() 
            ssim_epoch = ssim_epoch + ssim.cpu().item()
            unet_epoch = unet_epoch + unet_loss.cpu().item()
            i=i+1
        
        print()
        mse_epoch = mse_epoch/i
        ssim_epoch = ssim_epoch/i
        unet_epoch = unet_epoch/i
        graph_gloss.append(ssim_epoch)
        print("mse: + "+str(mse_epoch) + " | ssim: "+ str(ssim_epoch)+ " | unet:"+str(unet_epoch))
        print()

        if not os.path.exists(dir_for_output_weights):
            os.mkdir(dir_for_output_weights)
        
        path_of_generator_weight = os.path.join(dir_for_output_weights, 'generator_'+str(epoch+1)+'.pth')  #path for storing the weights of genertaor
        path_of_discriminator_weight = os.path.join(dir_for_output_weights, 'discriminator_'+str(epoch+1)+'.pth')  #path for storing the weights of discriminator
        DUNet.save_weight(path_of_generator_weight,path_of_discriminator_weight)


train(starting_epoch, ending_epoch) 