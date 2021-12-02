from libraries import *
from DU_Net import DU_Net

graph_gloss = []
input_unet_channel = 3
output_unet_channel = 3
input_dis_channel = 3
max_epochs = 100
DUNet = DU_Net(input_unet_channel ,output_unet_channel ,input_dis_channel).cuda()


# path_of_generator_weight = 'pretrained_weights/outdoor/generator.pth'  #path where the weights of genertaor are stored
# path_of_discriminator_weight = 'pretrained_weights/outdoor/discriminator.pth'  #path where the weights of discriminator are stored

path_of_generator_weight = 'weights_211201/generator_150.pth'  #path where the weights of genertaor are stored
path_of_discriminator_weight = 'weights_211201/discriminator_150.pth'  #path where the weights of discriminator are stored

DUNet.load(path_of_generator_weight,path_of_discriminator_weight)


def to_tensor(img):
    img_t = F6.to_tensor(img).float()
    return img_t

def postprocess(img):
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()


path_of_test_hazy_images = 'O-Haze/train/haze/*'
path_for_resultant_dehaze_images = 'results/train_211201_150/'
os.mkdir(path_for_resultant_dehaze_images)
image_paths_test_hazy=glob.glob(path_of_test_hazy_images)

for i in range(len(image_paths_test_hazy)):
    haze_image = cv2.imread(image_paths_test_hazy[i])
    haze_image = Img.fromarray(haze_image)
    haze_image = haze_image.resize((512,512), resample=PIL.Image.BICUBIC)
    haze_image = np.array(haze_image)
    haze_image = cv2.cvtColor(haze_image, cv2.COLOR_BGR2YCrCb)
    haze_image = to_tensor(haze_image).cuda()
    haze_image = haze_image.reshape(1,3,512,512)

    dehaze_image = DUNet.predict(haze_image) 
    
    dehaze_image = postprocess(dehaze_image)[0]
    dehaze_image = dehaze_image.cpu().detach().numpy()
    dehaze_image = dehaze_image.astype('uint8')
    dehaze_image = dehaze_image.reshape(512,512,3)
    dehaze_image = cv2.cvtColor(dehaze_image, cv2.COLOR_YCrCb2BGR)
    print(path_for_resultant_dehaze_images+image_paths_test_hazy[i].split('_')[0][-2:]+'.png')
    cv2.imwrite(path_for_resultant_dehaze_images+image_paths_test_hazy[i].split('_')[0][-2:]+'.png', dehaze_image)


# class EdgeAccuracy(nn.Module):
#     """
#     Measures the accuracy of the edge map
#     """
#     def __init__(self, threshold=0.5):
#         super().__init__()
#         self.threshold = threshold

#     def __call__(self, inputs, outputs):
#         labels = (inputs > self.threshold)
#         outputs = (outputs > self.threshold)

#         relevant = torch.sum(labels.float())
#         selected = torch.sum(outputs.float())

#         if relevant == 0 and selected == 0:
#             return 1, 1

#         true_positive = ((outputs == labels) * labels).float()
#         recall = torch.sum(true_positive) / (relevant + 1e-8)
#         precision = torch.sum(true_positive) / (selected + 1e-8)

#         return precision, recall


# class PSNR(nn.Module):
#     def __init__(self, max_val=0):
#         super().__init__()

#         base10 = torch.log(torch.tensor(10.0))
#         max_val = torch.tensor(max_val).float()

#         self.register_buffer('base10', base10)
#         self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

#     def __call__(self, a, b):
#         mse = torch.mean((a.float() - b.float()) ** 2)
    
#         if mse == 0:
#             return 0

#         return 1.0 / mse


# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
#     return gauss/gauss.sum()


# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window


# def _ssim(img1, img2, window, window_size, channel, size_average = True):
#     mu1 = F9.conv2d(img1, window, padding = window_size//2, groups = channel)
#     mu2 = F9.conv2d(img2, window, padding = window_size//2, groups = channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1*mu2

#     sigma1_sq = F9.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
#     sigma2_sq = F9.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
#     sigma12 = F9.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

#     C1 = 0.01**2
#     C2 = 0.03**2

#     ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)


# class SSIM(torch.nn.Module):
#     def __init__(self, window_size = 11, size_average = True):
#         super(SSIM, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 1
#         self.window = create_window(window_size, self.channel)

#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()

#         if channel == self.channel and self.window.data.type() == img1.data.type():
#             window = self.window
#         else:
#             window = create_window(self.window_size, channel)
            
#             if img1.is_cuda:
#                 window = window.cuda(img1.get_device())
#             window = window.type_as(img1)
            
#             self.window = window
#             self.channel = channel


#         return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# def ssim(img1, img2, window_size = 11, size_average = True):
#     (_, channel, _, _) = img1.size()
#     window = create_window(window_size, channel)
    
#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)
    
#     return _ssim(img1, img2, window, window_size, channel, size_average)


# ssim = SSIM(window_size = 11)
# psnr = PSNR()
# psnr_val = 0
# psnr_val = 0.0
# final_ssim = 0

# path_of_test_hazy_images = 'O-Haze/test/haze/*.png'
# path_of_test_gt_images = 'O-Haze/test/gt/*.png'
# path_for_resultant_dehaze_images = 'test/result/'

# image_paths_test_hazy=glob.glob(path_of_test_hazy_images)
# image_paths_test_gt=glob.glob(path_of_test_gt_images)

# for i in range(len(image_paths_test_hazy)):
#     im1 = cv2.imread(image_paths_GT[i])
#     im1 = Img.fromarray(im1)
#     im1 = im1.resize((512,512), resample=PIL.Image.BICUBIC)
#     im1 = np.array(im1)
#     im2 = cv2.imread('Results/experiment yrcrcb/new/outdoor/' + str(i+50)+'.png')

#     im1 = to_tensor(im1).reshape(1,3,512,512)
#     im2 = to_tensor(im2).reshape(1,3,512,512)
    
#     psnr_val = psnr(im1, im2)
#     final_psnr = final_psnr + 10*np.log10((psnr_val))
#     final_ssim = final_ssim + ssim(im1, im2)


# print(final_ssim/5.0, final_psnr/5.0)
